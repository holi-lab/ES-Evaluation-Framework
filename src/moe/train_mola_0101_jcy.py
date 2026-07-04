#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gc
from dataclasses import dataclass
from itertools import product
from typing import Dict, Any, List
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.optim import AdamW

import torch
#torch.set_printoptions(threshold=torch.inf)

import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
import wandb


# ============================================================
# 기본 설정
# ============================================================

# model_id = ""  # <-- SFT 끝난 체크포인트로 바꿔서 쓰면 됨
# train_file_path = ""
# val_file_path = ""
# base_output_dir = ""
# os.makedirs(base_output_dir, exist_ok=True)

# 폭포 버전

# 지금은 base llama에 학습
# 그 다음에 sft merge 모델로도 해보기
model_id = ""  # <-- SFT 끝난 체크포인트로 바꿔서 쓰면 됨
#train_file_path = ""
train_file_path = ""
#val_file_path = ""
val_file_path = ""
base_output_dir = ""
wandb_project = ""

if base_output_dir:
    os.makedirs(base_output_dir, exist_ok=True)


def require_config(name: str, value):
    if not value:
        raise ValueError(f"Please set `{name}` before running this script.")
    return value

# 실험 스케줄 (필요한 것만 남기면 해당 조합만 실행)
LR_LIST: List[float] = [2e-5] # -> 나중에 1e-5 혹은 2e-5 외 다른걸로 더 돌려보기
NUM_EXPERTS_LIST: List[int] = [4] # [5, 4, 3] # [3, 4, 5]

# 나머지 sweep 파라미터들도 동일한 방식으로 리스트로 관리
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


# preset_name -> module 조합을 등록하고, TARGET_MODULE_PRESET_ORDER로 어떤 조합을 돌릴지 결정
# 기본은 전체 다 포함
TARGET_MODULE_PRESETS: Dict[str, List[str]] = {
    "full_linear": DEFAULT_TARGET_MODULES,
    "attn_full": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_full": ["gate_proj", "up_proj", "down_proj"],
    "attn_qv": ["q_proj", "v_proj"],
    "mlp_gate_down": ["gate_proj", "down_proj"],
    "mixed_qv_down": ["q_proj", "v_proj", "down_proj"],
    "mixed_ko_up": ["k_proj", "o_proj", "gate_proj", "up_proj"],
}
# 러닝에 사용할 순서는 필요에 따라 자유롭게 선택
TARGET_MODULE_PRESET_ORDER: List[str] = ["full_linear"]

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# MoLA 하이퍼파라미터
ROUTER_HIDDEN_DIM = 128         # router 내부 hidden
ROUTER_NUM_RESID = 3            # 논문: residual blocks 3개
ALPHA_PROJ_DIM = 64             # TwD용 φ(α) 차원 (ω의 차원)
_default_target_key = TARGET_MODULE_PRESET_ORDER[0] if TARGET_MODULE_PRESET_ORDER else next(iter(TARGET_MODULE_PRESETS))
TARGET_MODULES = TARGET_MODULE_PRESETS[_default_target_key]

TEMP_TWD = 1.0                  # temperature τ

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 데이터 로딩 & 전처리
# ============================================================

def estimate_llama_dense_total_params(config) -> int | None:
    """
    Estimate original dense (non-quantized) total params from Llama config.
    Returns None if required fields are missing.
    """
    required = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]
    if any(not hasattr(config, k) for k in required):
        return None

    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    n_layers = int(config.num_hidden_layers)
    n_heads = int(config.num_attention_heads)
    vocab = int(config.vocab_size)
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))

    if hidden % n_heads != 0:
        return None

    head_dim = hidden // n_heads
    kv_hidden = n_kv_heads * head_dim

    attn_has_bias = bool(getattr(config, "attention_bias", False))
    mlp_has_bias = bool(getattr(config, "mlp_bias", False))
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    # attention projections
    q = hidden * hidden + (hidden if attn_has_bias else 0)
    k = hidden * kv_hidden + (kv_hidden if attn_has_bias else 0)
    v = hidden * kv_hidden + (kv_hidden if attn_has_bias else 0)
    o = hidden * hidden + (hidden if attn_has_bias else 0)

    # mlp projections
    gate = hidden * inter + (inter if mlp_has_bias else 0)
    up = hidden * inter + (inter if mlp_has_bias else 0)
    down = inter * hidden + (hidden if mlp_has_bias else 0)

    # RMSNorm weights only (input + post-attn per layer)
    layer_norms = hidden + hidden

    per_layer = q + k + v + o + gate + up + down + layer_norms

    embed = vocab * hidden
    final_norm = hidden
    lm_head = 0 if tie_word_embeddings else hidden * vocab

    return embed + (per_layer * n_layers) + final_norm + lm_head

def load_jsonl_file(file_path: str) -> Dataset:
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # data = data[:4000]  # 샘플링 (원하면 제거)
    return Dataset.from_list(data)


def extract_seeker_features_from_system(system_content: str) -> str:
    """
    system 메시지 content에서
    <seeker_features> ... </seeker_features> 사이의 텍스트만 추출.
    태그는 제거한 채 내부 텍스트만 반환.
    """
    start_tag = "<seeker_features>"
    end_tag = "</seeker_features>"

    start = system_content.find(start_tag)
    end = system_content.find(end_tag)

    if start == -1 or end == -1 or end <= start:
        return ""  # 없는 경우는 빈 문자열

    inner = system_content[start + len(start_tag):end]
    return inner.strip()


def formatting_prompts_func(examples, tokenizer_instance: AutoTokenizer):
    """
    - full_text: 기존 SFT용 input (system + user + assistant)
    - feature_text: seeker_features 블록만 추출
    - feature_emb: (사전 계산) 배치 각 샘플별 1:1 전달  ✅ FIX
    """
    full_texts: List[str] = []
    feature_texts: List[str] = []
    feature_embs: List[List[float]] = []

    has_emb = "feature_emb" in examples
    # 배치 길이 기준 반복
    for i, messages in enumerate(examples["messages"]):
        # 1) full_text
        if not messages or messages[-1].get("role") != "assistant":
            full_texts.append("")
            feature_texts.append("")
            # 길이 맞추기 위해 emb도 추가 (원본 있으면 그대로, 없으면 0 벡터/빈 리스트)
            feature_embs.append(examples["feature_emb"][i] if has_emb else [])
            continue

        full_text = tokenizer_instance.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_texts.append(full_text)

        # 2) feature_text
        system_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
                break
        feat_text = extract_seeker_features_from_system(system_msg.get("content", "")) if system_msg else ""
        feature_texts.append(feat_text)

        # 3) feature_emb  ✅ 각 샘플 고유 임베딩 유지
        feature_embs.append(examples["feature_emb"][i] if has_emb else [])

    return {
        "full_text": full_texts,
        "feature_text": feature_texts,
        "feature_emb": feature_embs,  # ← 배치 길이와 반드시 동일
    }


def tokenize_function(examples, tokenizer_instance: AutoTokenizer):
    """
    - full_text -> input_ids, attention_mask
    - feature_text -> feat_input_ids, feat_attention_mask
    - feature_text 원문도 그대로 보존 (TwD 라벨생성/디버깅용)
    """
    full_enc = tokenizer_instance(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
    )
    feat_enc = tokenizer_instance(
        examples["feature_text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # feature 텍스트는 짧음
    )

    enc = {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        # ↓ TwD 라벨/로그 생성을 위해 그대로 넘김
        "feature_text": examples["feature_text"],
        "feature_emb": examples["feature_emb"],
    }
    return enc


# ============================================================
# DataCollator
# ============================================================
def custom_mask_labels(batch, tokenizer):
    """
    마지막 <|start_header_id|>assistant<|end_header_id|> '헤더 완료 지점' 이후만 label 유지.
    그 전은 전부 -100으로 마스킹.
    """
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    sid = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    eid = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    pad_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("[PAD]")

    if sid is None or eid is None or len(assistant_ids) == 0:
        batch["labels"] = torch.full_like(labels, -100)
        return batch

    pattern = [sid] + assistant_ids + [eid]
    plen = len(pattern)

    for i in range(labels.size(0)):
        ids = input_ids[i].tolist()
        hits = []
        for j in range(0, len(ids) - plen + 1):
            if ids[j:j+plen] == pattern:
                hits.append(j)

        if len(hits) == 0:
            labels[i, :] = -100
            continue

        start = hits[-1] + plen
        labels[i, :start] = -100

    batch["labels"] = labels

    # ===== 디버깅용 출력 (패딩 제외) =====
    # for i in range(labels.size(0)):
    #     this_labels = labels[i]
    #     this_input = input_ids[i]

    #     # 패딩이 아닌 부분만 계산
    #     valid_mask = (this_input != pad_id)
    #     valid_labels = this_labels[valid_mask]

    #     n_valid = valid_labels.numel()
    #     n_unmasked = (valid_labels != -100).sum().item()
    #     ratio = n_unmasked / n_valid if n_valid > 0 else 0.0

    #     print(f"\n[Sample {i}] valid tokens: {n_valid}, unmasked: {n_unmasked}, ratio: {ratio:.4f}")

    #     # (선택) not-masked text 확인
    #     not_masked_idx = (this_labels != -100).nonzero(as_tuple=True)[0].tolist()
    #     if not_masked_idx:
    #         not_masked_text = tokenizer.decode(
    #             [this_input[j].item() for j in not_masked_idx],
    #             skip_special_tokens=False
    #         )
    #         print(f"Not masked input text: {not_masked_text}")

    return batch

import re

def parse_features(text: str) -> Dict[str, str]:
    """
    <seeker_features> 블록에서 5개 feature를 딕셔너리로 파싱.
    없는 값은 빈 문자열로 반환.
    """
    features = {
        "coping_strategy": "",
        "resistance_level": "",
        "engagement_level": "",
        "self_disclosure_level": "",
        "utterance_style": ""
    }

    for key in features.keys():
        pattern = rf"-\s*{key}\s*:\s*([^\n\r]+)"
        match = re.search(pattern, text)
        if match:
            features[key] = match.group(1).strip()

    return features


def _hash_label_strings_to_ids_multi(feature_texts: List[str]) -> Dict[str, torch.Tensor]:
    """
    feature_texts 전체 리스트를 받아 feature별로 해시 라벨 생성.
    반환값: {feature_name: torch.tensor([...])}
    """
    all_labels = {
        "coping_strategy": [],
        "resistance_level": [],
        "engagement_level": [],
        "self_disclosure_level": [],
        "utterance_style": []
    }

    for text in feature_texts:
        parsed = parse_features(text)
        for key, val in parsed.items():
            # 간단한 FNV 해시 (문자열 값 기준)
            h = 1469598103934665603
            for ch in val:
                h ^= ord(ch)
                h *= 1099511628211
                h &= (1 << 64) - 1
            all_labels[key].append(int(h % (10**9)))

    # 각 feature별 torch tensor로 반환
    out = {k: torch.tensor(v, dtype=torch.long) for k, v in all_labels.items()}
    return out



@dataclass
class MoLADataCollator:
    tokenizer: AutoTokenizer
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        lm_features = [
            {k: v for k, v in f.items()
             if k not in ["full_text", "feature_text", "feature_emb"]}  
            for f in features
        ]

        lm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=self.mlm)
        batch = lm_collator(lm_features)
        batch = custom_mask_labels(batch, self.tokenizer)

        # ★ precomputed embedding을 [B, D] 텐서로
        if "feature_emb" in features[0]:
            feat_emb = torch.tensor([f["feature_emb"] for f in features], dtype=torch.bfloat16)
            batch["feature_emb"] = feat_emb

        # ★ TwD용 라벨 (feature_text 해시)
        raw_feats: List[str] = [f.get("feature_text", "") for f in features]
        multi_labels = _hash_label_strings_to_ids_multi(raw_feats)
        for key, tensor in multi_labels.items():
            batch[f"feature_labels_{key}"] = tensor


        return batch



# ============================================================
# MoLA 모듈 정의
#   - MoLA_Linear: Router 없음. 레이어별 experts(A_i, B_i)만 보유.
#   - SharedRouter: 모델 전체에서 1개. α(sample-level) 생성.
# ============================================================

class ModeController:
    """전 레이어에서 공유하는 α 저장."""
    def __init__(self):
        self.shared_alpha: torch.Tensor | None = None


class MoLA_Linear(nn.Module):
    """
    base_linear (Llama의 q/k/v/o or gate/up/down)을 감싸는 MoLA 레이어.
    - Router 없음 (공유 Router가 따로 계산).
    - 각 레이어는 NUM_EXPERTS 개의 low-rank (A_i, B_i)만 가짐.
    - forward 시 controller.shared_alpha(B,E)를 읽어 mixture 업데이트.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        controller: ModeController,
        num_experts: int,
        rank: int,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.base = base_linear
        self.controller = controller

        # in/out feature
        if hasattr(base_linear, "in_features") and hasattr(base_linear, "out_features"):
            self.in_features = base_linear.in_features
            self.out_features = base_linear.out_features
        else:
            w = base_linear.weight
            self.out_features, self.in_features = w.shape

        self.num_experts = num_experts
        self.rank = rank

        # Experts (layer-local)
        self.As = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.in_features, rank, dtype=dtype)) for _ in range(num_experts)]
        )
        self.Bs = nn.ParameterList(
            [nn.Parameter(torch.zeros(rank, self.out_features, dtype=dtype)) for _ in range(num_experts)]
        )
        for A, B in zip(self.As, self.Bs):
            nn.init.kaiming_uniform_(A, a=2**0.5)
            nn.init.zeros_(B)

        # base 동결
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D] or [B,D]
        y = base(x) + Σ α_i ( x @ A_i @ B_i )
        """
        y_base = self.base(x)

        # normalize shape
        squeezed = False
        if x.dim() == 2:
            x_ = x.unsqueeze(1)  # [B,1,D]
            squeezed = True
        else:
            x_ = x  # [B,T,D]

        B = x_.shape[0]

        alpha = self.controller.shared_alpha
        assert alpha is not None, "shared_alpha is None. Call model.compute_alpha(...) before forward."
        # alpha: [B, E]
        # dtype align (updates 계산은 입력 dtype로)
        alpha = alpha.to(x_.dtype)

        updates = torch.zeros_like(y_base, dtype=x_.dtype, device=x_.device)

        for i in range(self.num_experts):
            XA = torch.matmul(x_, self.As[i])
            XAB = torch.matmul(XA, self.Bs[i])
            a = alpha[:, i].view(B, 1, 1)
            # XAB.mul_(a)           # ✅ in-place
            # updates.add_(XAB)     # ✅ in-place
            updates = updates + XAB * a


        if squeezed:
            updates = updates.squeeze(1)  # [B,out]

        return y_base + updates


def inject_mola_linear(
    model: nn.Module,
    controller: ModeController,
    target_modules: List[str],
    num_experts: int,
    mola_rank: int,
) -> nn.Module:
    """
    Llama 내부의 target Linear 모듈을 MoLA_Linear로 교체.
    """
    def _replace(module: nn.Module):
        for name, child in list(module.named_children()):
            if name in target_modules and isinstance(child, nn.Module) and hasattr(child, "weight"):
                mola = MoLA_Linear(child, controller, num_experts=num_experts, rank=mola_rank)
                setattr(module, name, mola)
            else:
                _replace(child)

    _replace(model)
    return model


# -------------------- Shared Router -------------------- #

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, dtype=dtype)
        self.norm = nn.LayerNorm(dim, dtype=dtype)

    def forward(self, x):
        # x: [B, D], must be bfloat16 as well
        h = self.fc2(self.act(self.fc1(x)))
        return self.norm(x + h)



class SharedRouter(nn.Module):
    """
    논문: 3 residual blocks φ + gate g → α (B,E)
    입력 x는 feature token의 임베딩 평균을 사용 (Transformer 전체를 돌리지 않음).
    """
    def __init__(self, embed_dim: int, num_experts: int, hidden: int = ROUTER_HIDDEN_DIM, n_blocks: int = ROUTER_NUM_RESID, dtype=torch.bfloat16):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualMLP(embed_dim, hidden, dtype=dtype) for _ in range(n_blocks)])
        self.gate = nn.Linear(embed_dim, num_experts, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] pooled embedding
        h = x
        for blk in self.blocks:
            h = blk(h)
        logits = self.gate(h)          # [B, E]
        alpha = F.softmax(logits, dim=-1)
        return alpha


# ============================================================
# MoLA 모델 래퍼 (Shared Router + α projector for TwD)
# ============================================================

class MoLAModel(nn.Module):
    """
    - base_model: AutoModelForCausalLM (양자화된 Llama)
    - controller: 모든 MoLA_Linear가 참조하는 shared_alpha 저장
    - shared_router: 1개
    - alpha_projector: TwD용 φ(α) → ω
    """

    def __init__(self, base_model, controller, num_experts, feature_dim, dtype=torch.bfloat16):
        super().__init__()
        self.base_model = base_model
        self.controller = controller

        self.shared_router = SharedRouter(feature_dim, num_experts)

        # TwD projector φ: α → ω
        self.alpha_projector = nn.Sequential(
            nn.Linear(num_experts, ALPHA_PROJ_DIM, dtype=dtype),
            nn.GELU(),
            nn.Linear(ALPHA_PROJ_DIM, ALPHA_PROJ_DIM, dtype=dtype),
        )

        # 학습 파라미터 확인용
        self._frozen = True

    # ---- α 계산 (한 번만) ----

    def compute_alpha_from_features(self, feature_emb: torch.Tensor):
        """
        feature_emb: [B, D] precomputed embedding vectors
        """
        #print("feat_emb batch var:", feature_emb.var(dim=1).mean().item())
        
        alpha = self.shared_router(feature_emb.to(self.base_model.device))
        self.controller.shared_alpha = alpha
        
        return alpha


    # ---- LM forward (single pass) ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        """
        shared_alpha가 이미 설정되어 있다는 전제 하에 LM 1-pass.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs


# ============================================================
# TwD Loss
# ============================================================

def compute_twd_loss_neg_only(
    alpha: torch.Tensor,
    labels_dict: Dict[str, torch.Tensor],
    projector: nn.Module,
    tau: float = 0.2,
) -> torch.Tensor:
    B, E = alpha.shape
    device = alpha.device

    # omega
    omega = projector(alpha)
    omega = torch.nn.functional.normalize(omega, dim=-1)  # ★ 매우 중요
    sim = torch.matmul(omega, omega.t()) / tau

    eye = torch.eye(B, device=device, dtype=torch.bool)
    total_loss = alpha.new_tensor(0.0)
    n_feat = len(labels_dict)

    for feat_name, labels in labels_dict.items():
        losses = []

        # negative: feature가 다른 pair
        neg = (labels.unsqueeze(0) != labels.unsqueeze(1)) & (~eye)

        for i in range(B):
            neg_mask = neg[i]
            if neg_mask.sum() == 0:
                continue

            # log-sum-exp over negatives
            loss_i = torch.logsumexp(sim[i][neg_mask], dim=0)
            losses.append(loss_i)

        if len(losses) > 0:
            loss_feat = torch.stack(losses).mean()
            total_loss = total_loss + loss_feat

    return total_loss / n_feat

def pairwise_sq_l2(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, D]
    return: [B, B]  where M[i,j] = ||x_i - x_j||^2
    """
    # ||x||^2
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [B,1]

    # ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())

    return torch.clamp(dist, min=0.0)

def build_gold_alpha_from_feature0(
    feature_emb: torch.Tensor,
    num_experts: int,
    threshold: float = 0.5,
):
    """
    feature_emb: [B, D]
    return: gold_alpha [B, E] (one-hot)
    """
    assert num_experts >= 2, "Need at least 2 experts"

    B = feature_emb.size(0)
    device = feature_emb.device

    gold = torch.zeros(B, num_experts, device=device)

    mask = feature_emb[:, 0] > threshold   # [B] bool

    gold[mask, 0] = 1.0        # expert 0
    gold[~mask, 1] = 1.0       # expert 1

    return gold


def alpha_supervised_ce_loss(alpha: torch.Tensor, gold_alpha: torch.Tensor):
    """
    alpha: [B, E] (softmax output)
    gold_alpha: [B, E] (one-hot)
    """
    log_alpha = torch.log(alpha + 1e-12)
    loss = -(gold_alpha * log_alpha).sum(dim=1).mean()
    return loss

def feature_alpha_sep_loss(alpha, feature_vec, margin=0.2):
    # pairwise squared distances
    Da = pairwise_sq_l2(alpha)
    Df = pairwise_sq_l2(feature_vec)

    # scale normalization (중요)
    Da = Da / (Da.mean().detach() + 1e-6)
    Df = Df / (Df.mean().detach() + 1e-6)

    # hinge-style separation
    loss = torch.relu(Df - Da + margin)

    # remove self-pairs
    mask = ~torch.eye(Da.size(0), device=Da.device, dtype=torch.bool)
    return loss[mask].mean()

def debug_sep_stats(alpha, feat, margin=0.2):
    Da = pairwise_sq_l2(alpha)
    Df = pairwise_sq_l2(feat)

    B = Da.size(0)
    mask = ~torch.eye(B, device=Da.device, dtype=torch.bool)

    Da_m = Da[mask]
    Df_m = Df[mask]

    # normalized (현재 loss랑 동일)
    Da_n = Da / (Da.mean().detach() + 1e-6)
    Df_n = Df / (Df.mean().detach() + 1e-6)

    Da_n_m = Da_n[mask]
    Df_n_m = Df_n[mask]

    x = (Df_n_m - Da_n_m + margin)

    stats = {
        "Df_mean": Df_m.mean().item(),
        "Da_mean": Da_m.mean().item(),
        "DfN_mean": Df_n_m.mean().item(),
        "DaN_mean": Da_n_m.mean().item(),
        "hinge_pos_ratio": (x > 0).float().mean().item(),
        "x_mean": x.mean().item(),
        "alpha_var": alpha.var(dim=0).mean().item(),
        "feat_var": feat.var(dim=0).mean().item(),
    }
    return stats

from transformers import TrainerCallback

class RouterParamCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        # with torch.no_grad():
        #     bias = model.shared_router.gate.bias
        #     print(
        #         f"[STEP {state.global_step}] "
        #         f"gate.bias mean={bias.mean().item():.6f}, "
        #         f"std={bias.std().item():.6f}"
        #     )

def split_router_expert_params(model):
    router_params = []
    expert_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("shared_router"):
            router_params.append(p)
        elif "As." in name or "Bs." in name:
            expert_params.append(p)
        else:
            # alpha_projector 등 (원하면 router 쪽에 합쳐도 됨)
            router_params.append(p)

    return router_params, expert_params


def compute_twd_loss_multi(
    alpha: torch.Tensor,
    labels_dict: Dict[str, torch.Tensor],
    projector: nn.Module,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    α: [B, E]
    labels_dict: {
        'coping_strategy': Tensor([B]),
        'resistance_level': Tensor([B]),
        ...
    }
    projector: α → ω
    """

    B, E = alpha.shape
    device = alpha.device

    # 공통 ω 계산 (한 번만)
    omega = projector(alpha)               # [B, D_ω]
    sim = torch.matmul(omega, omega.t()) / tau   # [B, B]
    total_loss = alpha.new_tensor(0.0)
    
    n_feat = len(labels_dict)
    #print("alpha: ", alpha, "sim: ", sim)

    for feat_name, labels in labels_dict.items():
        # self-sim 제외
        eye = torch.eye(B, device=device, dtype=torch.bool)
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye)
        neg = ~eye

        losses = []
        for i in range(B):
            pos_mask = pos[i]
            neg_mask = neg[i]
            if pos_mask.sum() == 0:
                continue
            num = torch.exp(sim[i][pos_mask]).sum()
            den = torch.exp(sim[i][neg_mask]).sum()
            losses.append(-torch.log(num / den + 1e-12))

        if len(losses) > 0:
            loss_feat = torch.stack(losses).mean()
            total_loss = total_loss + loss_feat

    return total_loss / n_feat

    # # 수정사항
    # omega = projector(alpha)  # [B, D]
    # omega = F.normalize(omega, p=2, dim=-1)  # 안정적으로 cosine 계산
    

    # # cosine sim: [B,B] in [-1,1]
    # sim = omega @ omega.t()
    # dist = 1.0 - sim  # cosine distance in [0,2]

    # omega = projector(alpha).float()  # fp32
    # sq = (omega ** 2).sum(dim=1, keepdim=True)  # [B,1]
    # dist = torch.sqrt(
    #     sq - 2 * (omega @ omega.t()) + sq.t() + 1e-8
    # )
    # loss_mat = F.relu(tau - dist)



    # eye = torch.eye(B, device=device, dtype=torch.bool)

    # total = alpha.new_tensor(0.0)
    # n_feat_used = 0

    # for feat_name, labels in labels_dict.items():
    #     # "다른 라벨" pair만 선택 (i!=j)
    #     diff = (labels.unsqueeze(0) != labels.unsqueeze(1)) & (~eye)

    #     if diff.any():
    #         # hinge: dist >= margin 되게
    #         # dist가 margin보다 작으면 penalty
    #         loss_mat = F.relu(tau - dist)
    #         loss_feat = loss_mat[diff].mean()
    #         total = total + loss_feat
    #         n_feat_used += 1


    # if n_feat_used == 0:
    #     return total  # 0.0
    # return total / n_feat_used



# ============================================================
# Trainer (TwD delayed accumulation version)
# ============================================================

class MoLATrainer(Trainer):
    def __init__(self, *args, lambda_twd: float, **kwargs):
        self.lambda_twd = lambda_twd
        super().__init__(*args, **kwargs)
        self.twd_buffer_alpha = []
        self.twd_buffer_labels = []
        self.accum_steps = self.args.gradient_accumulation_steps
        self._micro = 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        router_params, expert_params = split_router_expert_params(self.model)

        self.optimizer = AdamW(
            [
                {"params": router_params, "lr": self.args.learning_rate}, # embedding이면 1e-6, ordinal vector면 1e-3 쓰자
                {"params": expert_params, "lr": 2e-5},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        return self.optimizer

    def compute_loss(self, model: MoLAModel, inputs: Dict[str, Any], return_outputs=False, **kwargs):
        feature_emb = inputs.pop("feature_emb")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        labels_dict = {
            "coping_strategy": inputs.pop("feature_labels_coping_strategy"),
            "resistance_level": inputs.pop("feature_labels_resistance_level"),
            "engagement_level": inputs.pop("feature_labels_engagement_level"),
            "self_disclosure_level": inputs.pop("feature_labels_self_disclosure_level"),
            "utterance_style": inputs.pop("feature_labels_utterance_style"),
        }


        alpha = model.compute_alpha_from_features(feature_emb)
        #print("alpha: ", alpha)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss

        #alpha_loss = feature_alpha_sep_loss(alpha, feature_emb)
        alpha_loss = compute_twd_loss_multi(
            alpha=alpha,
            labels_dict=labels_dict,
            projector=model.alpha_projector,
            tau=TEMP_TWD,
        )

        loss = lm_loss + self.lambda_twd * alpha_loss
        # loss = self.lambda_twd * alpha_loss
        
        if self.state.global_step is not None:
            self.log({
                "alpha_loss": alpha_loss.detach().item(),
                "lm_loss": lm_loss.detach().item(),
                "total_loss": loss.detach().item(),
            })

        return (loss, outputs) if return_outputs else loss
    
    # def compute_loss(self, model: MoLAModel, inputs: Dict[str, Any], return_outputs=False, **kwargs):
    #     # --- inputs ---
    #     feature_emb = inputs.pop("feature_emb")        # [B, F]
    #     input_ids = inputs.get("input_ids")
    #     attention_mask = inputs.get("attention_mask")
    #     labels = inputs.get("labels")

    #     # --- alpha ---
    #     alpha = model.compute_alpha_from_features(feature_emb)  # [B, E]

    #     # --- LM loss ---
    #     lm_loss = 0
    #     outputs = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels
    #     )
    #     lm_loss = outputs.loss

    #     # --- gold alpha ---
    #     gold_alpha = build_gold_alpha_from_feature0(
    #         feature_emb,
    #         num_experts=alpha.size(1),
    #         threshold=0.5
    #     )

    #     # --- supervised alpha loss ---
    #     alpha_loss = alpha_supervised_ce_loss(alpha, gold_alpha)

    #     # (처음엔 LM loss 꺼도 됨)
    #     #loss = lm_loss + self.lambda_twd * alpha_loss
    #     loss = alpha_loss

    #     # --- logging ---
    #     if self.state.global_step is not None:
    #         self.log({
    #             "alpha_loss": alpha_loss.detach().item(),
    #             "lm_loss": lm_loss.detach().item(),
    #             "total_loss": loss.detach().item(),
    #         })

    #     return (loss, outputs) if return_outputs else loss


# ============================================================
# 학습 루틴
# ============================================================

def format_lr_for_name(lr: float) -> str:
    # readable string (e.g., 1e-5 or 0.025)
    if lr < 1e-2:
        s = f"{lr:.0e}"
        mantissa, exponent = s.split("e")
        exp_sign = ""
        exp_val = exponent
        if exponent.startswith(("+", "-")):
            exp_sign = exponent[0]
            exp_val = exponent[1:]
        exp_val = exp_val.lstrip("0") or "0"
        if exp_sign == "+":
            exp_sign = ""
        return f"{mantissa}e{exp_sign}{exp_val}"
    else:
        s = f"{lr:.3f}".rstrip("0").rstrip(".")
        return s

def format_ne_for_name(num_experts: int) -> str:
    return f"ne{num_experts}"

def format_rank_for_name(rank: int) -> str:
    return f"rank{rank}"

def format_lambda_for_name(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return f"twd{s}"

def format_target_for_name(modules: List[str], preset_name: str | None = None) -> str:
    if preset_name:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in preset_name)
        return f"tm_{safe_name}"
    joined = "-".join(modules)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in joined)
    return f"tm_{safe}"

def train_model(
    learning_rate: float,
    num_experts: int,
    lambda_twd: float,
    mola_rank: int,
    target_modules: List[str] | None = None,
    target_preset_name: str | None = None,
):
    require_config("model_id", model_id)
    require_config("train_file_path", train_file_path)
    require_config("val_file_path", val_file_path)
    require_config("base_output_dir", base_output_dir)
    require_config("wandb_project", wandb_project)

    base_run_ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M")

    target_modules = list(target_modules or TARGET_MODULES)
    if target_preset_name is None:
        for name, preset_modules in TARGET_MODULE_PRESETS.items():
            if preset_modules == target_modules:
                target_preset_name = name
                break

    lr_tag = format_lr_for_name(learning_rate)
    ne_tag = format_ne_for_name(num_experts)
    rank_tag = format_rank_for_name(mola_rank)
    lambda_tag = format_lambda_for_name(lambda_twd)
    target_tag = format_target_for_name(target_modules, target_preset_name)
    run_name = f"MOLA_{base_run_ts}_{ne_tag}_lr{lr_tag}_{rank_tag}_{lambda_tag}_{target_tag}"
    run_output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n=== [MoLA] Starting training (Shared Router + TwD) ===")
    print(f"Run name: {run_name}")
    print(f"Output dir: {run_output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num experts: {num_experts}")
    print(f"MoLA rank: {mola_rank}")
    print(f"TwD lambda: {lambda_twd}")
    if target_preset_name:
        print(f"Target module preset: {target_preset_name}")
    print(f"Target modules: {target_modules}")

    current_output_dir = run_output_dir
    os.makedirs(current_output_dir, exist_ok=True)

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    # raw 데이터 로드
    train_raw = load_jsonl_file(train_file_path)
    eval_raw = load_jsonl_file(val_file_path)

    # full_text / feature_text 생성
    processed_train = train_raw.map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
    ).filter(lambda x: x["full_text"] != "")

    processed_eval = eval_raw.map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
        remove_columns=eval_raw.column_names,
    ).filter(lambda x: x["full_text"] != "")

    # 토크나이즈 (+ feature_text pass-through)
    tokenized_train = processed_train.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
    )
    tokenized_eval = processed_eval.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
    )


    print(f"Train size: {len(tokenized_train)}, Eval size: {len(tokenized_eval)}")

    # 모델 로드 (양자화 Llama)
    if "base_model" in globals():
        try:
            base_model.cpu()
            del globals()["base_model"]
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=None,  # 단일 GPU 가정
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # backbone 동결
    for p in base_model.parameters():
        p.requires_grad = False

    # Controller & MoLA 주입
    controller = ModeController()
    base_model = inject_mola_linear(
        base_model,
        controller,
        target_modules=target_modules,
        num_experts=num_experts,
        mola_rank=mola_rank,
    )
    base_model.to(DEVICE)

    # MoLA 래퍼
    sample_feat = tokenized_train[0]["feature_emb"]
    feature_dim = len(sample_feat)

    model = MoLAModel(base_model, controller, num_experts=num_experts, feature_dim=feature_dim)
    model.to(DEVICE)

    # 학습 가능한 파라미터 수 출력
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_total = estimate_llama_dense_total_params(base_model.config)
    print(f"Total params: {total:,d}, Trainable params: {trainable:,d} ({100*trainable/total:.4f}%)")
    if dense_total is not None:
        print(f"Estimated original dense total params (Llama config): {dense_total:,d}")
    else:
        print("Estimated original dense total params (Llama config): unavailable")

    # TrainingArguments
    args = TrainingArguments(
        output_dir=current_output_dir,
        per_device_train_batch_size=3,         # TwD loss는 positive pair끼리 alpha를 끌어당기기 때문에 batch에 같은 label이 존재해야 함
        gradient_accumulation_steps=1,
        optim="adamw_torch",#"paged_adamw_8bit",
        learning_rate=learning_rate,                    # 어댑터/라우터/프로젝터만 학습
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="steps",  # save a checkpoint periodically
        save_steps=5000,
        eval_steps=5000,
        eval_strategy="steps", 
        group_by_length=True,
        #report_to="none",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    wandb_config = {
        # TrainingArguments 전체를 dict로 변환
        **args.to_dict(),
        **bnb_config.to_dict(),
        
        # 추가 설정들
        "train_type": "CL",
        "base_data_path": "",
        "train_file": train_file_path,
        "val_file": val_file_path,
        "base_output_dir": base_output_dir,
        "run_name": run_name,
        "model_id": model_id,
        "NUM_EXPERTS": num_experts,
        "MOLA_RANK": mola_rank,
        "ROUTER_HIDDEN_DIM": ROUTER_HIDDEN_DIM,
        "ROUTER_NUM_RESID": ROUTER_NUM_RESID,
        "ALPHA_PROJ_DIM": ALPHA_PROJ_DIM,
        "TARGET_MODULES": target_modules,
        "TARGET_MODULE_PRESET": target_preset_name,
        "LAMBDA_TWD": lambda_twd,
        "TEMP_TWD": TEMP_TWD
    }

    wandb.init(
        project=wandb_project,
        name=run_name,
        config=wandb_config,
        reinit=True,  # sweep 내 각 설정마다 독립적인 run 생성
    )

    # wandb config를 JSON 파일로 저장
    config_save_path = os.path.join(run_output_dir, "wandb_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(wandb_config, f, indent=2, ensure_ascii=False)
    print(f"Saved wandb config to {config_save_path}")

    data_collator = MoLADataCollator(tokenizer=tokenizer, mlm=False)

    trainer = MoLATrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        lambda_twd=lambda_twd,
        callbacks=[RouterParamCallback()],   # 👈 추가
    )

    opt = trainer.create_optimizer()
    print("Router LR:", trainer.optimizer.param_groups[0]["lr"])
    print("Expert LR:", trainer.optimizer.param_groups[1]["lr"])
    print("Router params:", len(trainer.optimizer.param_groups[0]["params"]))
    print("Expert params:", len(trainer.optimizer.param_groups[1]["params"]))

    print("Start MoLA training …")
    trainer.train()
    print("Training done.")

    # 모델 저장 (backbone+MoLA 전체 저장)
    trainer.save_model(current_output_dir)
    print(f"Model saved to: {current_output_dir}")

    # 다음 설정이 동일 run을 재사용하지 않도록 명시적으로 종료
    wandb.finish()

    model.cpu()
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    # LAMBDA_TWD_LIST, NUM_EXPERTS_LIST, LR_LIST 모두 argument로 받기
    import argparse
    parser = argparse.ArgumentParser() 
    # list 더하기 형태로 받기
    parser.add_argument("--lambda_twd_list", nargs="+", type=float, default=[5.0])
    parser.add_argument("--num_experts_list", nargs="+", type=int, default=[4])
    parser.add_argument("--lr_list", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--mola_rank_list", nargs="+", type=int, default=[4])
    args = parser.parse_args()
    LAMBDA_TWD_LIST = args.lambda_twd_list
    NUM_EXPERTS_LIST = args.num_experts_list
    LR_LIST = args.lr_list
    MOLA_RANK_LIST = args.mola_rank_list

    sweep_combinations = list(product(LAMBDA_TWD_LIST, NUM_EXPERTS_LIST, LR_LIST))
    for target_key in TARGET_MODULE_PRESET_ORDER:
        target_modules = TARGET_MODULE_PRESETS[target_key]
        for mola_rank in MOLA_RANK_LIST:
            for lambda_twd, ne, lr in sweep_combinations:
                print(
                    f"\n\n########## START TM_PRESET={target_key}, "
                    f"Rank={mola_rank}, Lambda={lambda_twd}, NE={ne}, LR={lr} ##########"
                )
                train_model(
                    learning_rate=lr,
                    num_experts=ne,
                    lambda_twd=lambda_twd,
                    mola_rank=mola_rank,
                    target_modules=target_modules,
                    target_preset_name=target_key,
                )

    print("\n=== All MoLA runs completed ===")
