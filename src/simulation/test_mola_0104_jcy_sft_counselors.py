import os

# os.environ.setdefault("TRANSFORMERS_NO_DYNAMIC_CACHE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, PeftConfig
import json
import gc
import re
import warnings
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import re
import warnings
import argparse
from typing import Any, Dict, List, Optional
from safetensors.torch import load_file
warnings.filterwarnings("ignore")

from safetensors import safe_open


# === MoLA 설정/유틸 (train_mola_new_w_twd.py에서 가져옴) ===
NUM_EXPERTS = 4 # 학습 시 개수랑 맞춰주기
MOLA_RANK = 4
ROUTER_HIDDEN_DIM = 128
ROUTER_NUM_RESID = 3
ALPHA_PROJ_DIM = 64
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def infer_feature_dim_from_testset(test_set_path: str) -> int:
    with open(test_set_path, "r", encoding="utf-8") as f:
        first = json.loads(next(f))
    assert "feature_emb" in first, "feature_emb not found in test set"
    return len(first["feature_emb"])

# ------------------------------
# MoLA inference patch helpers
# (keep minimal; do not touch existing helpers)
# ------------------------------


def _coerce_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        try:
            numeric_value = int(cleaned)
        except ValueError:
            try:
                numeric_value = int(float(cleaned))
            except ValueError:
                return None
        return numeric_value if numeric_value > 0 else None
    return None


def resolve_num_experts_from_checkpoint(checkpoint_path: str, fallback: int = NUM_EXPERTS) -> int:
    config_path = os.path.join(checkpoint_path, "wandb_config.json")
    if not os.path.exists(config_path):
        print(f"[INFO] wandb_config.json not found at {config_path}. Using default NUM_EXPERTS={fallback}.")
        return fallback

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as exc:
        print(f"[INFO] Failed to load {config_path} ({exc}). Using default NUM_EXPERTS={fallback}.")
        return fallback

    for key, value in config_data.items():
        normalized = key.strip().lower().replace(" ", "_")
        if normalized == "num_experts":
            resolved = _coerce_positive_int(value)
            if resolved is not None:
                print(f"[INFO] Resolved NUM_EXPERTS={resolved} from {config_path}.")
                return resolved
            print(f"[INFO] Invalid NUM_EXPERTS value '{value}' in {config_path}. Using default {fallback}.")
            return fallback

    print(f"[INFO] NUM_EXPERTS not found in {config_path}. Using default NUM_EXPERTS={fallback}.")
    return fallback


def resolve_mola_rank_from_checkpoint(checkpoint_path: str, fallback: int = MOLA_RANK) -> int:
    """
    wandb_config.json에서 MOLA_RANK를 읽어와 추론 시 rank를 자동 조정.
    """
    config_path = os.path.join(checkpoint_path, "wandb_config.json")
    if not os.path.exists(config_path):
        print(f"[INFO] wandb_config.json not found at {config_path}. Using default MOLA_RANK={fallback}.")
        return fallback

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as exc:
        print(f"[INFO] Failed to load {config_path} ({exc}). Using default MOLA_RANK={fallback}.")
        return fallback

    for key, value in config_data.items():
        normalized = key.strip().lower().replace(" ", "_")
        if normalized == "mola_rank":
            resolved = _coerce_positive_int(value)
            if resolved is not None:
                print(f"[INFO] Resolved MOLA_RANK={resolved} from {config_path}.")
                return resolved
            print(f"[INFO] Invalid MOLA_RANK value '{value}' in {config_path}. Using default {fallback}.")
            return fallback

    print(f"[INFO] MOLA_RANK not found in {config_path}. Using default MOLA_RANK={fallback}.")
    return fallback




class ModeController:
    """전 레이어에서 공유하는 α 저장."""

    def __init__(self):
        self.shared_alpha: Optional[torch.Tensor] = None


class MoLA_Linear(torch.nn.Module):
    """
    base_linear (Llama의 q/k/v/o or gate/up/down)을 감싸는 MoLA 레이어.
    Router 없음(공유 Router가 따로 계산).
    """

    def __init__(
        self,
        base_linear: torch.nn.Module,
        controller: ModeController,
        num_experts: int = NUM_EXPERTS,
        rank: int = MOLA_RANK,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.base = base_linear
        self.controller = controller

        if hasattr(base_linear, "in_features") and hasattr(base_linear, "out_features"):
            self.in_features = base_linear.in_features
            self.out_features = base_linear.out_features
        else:
            w = base_linear.weight
            self.out_features, self.in_features = w.shape

        self.num_experts = num_experts
        self.rank = rank

        self.As = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(self.in_features, rank, dtype=dtype)) for _ in range(num_experts)]
        )
        self.Bs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(rank, self.out_features, dtype=dtype)) for _ in range(num_experts)]
        )
        for A, B in zip(self.As, self.Bs):
            torch.nn.init.kaiming_uniform_(A, a=2**0.5)
            torch.nn.init.zeros_(B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.base(x)

        squeezed = False
        if x.dim() == 2:
            x_ = x.unsqueeze(1)
            squeezed = True
        else:
            x_ = x

        B = x_.shape[0]
        alpha = self.controller.shared_alpha
        assert alpha is not None, "shared_alpha is None. Call compute_alpha_from_features before forward."
        alpha = alpha.to(x_.dtype)

        updates = 0.0
        for i in range(self.num_experts):
            XA = torch.matmul(x_, self.As[i])
            XAB = torch.matmul(XA, self.Bs[i])
            a = alpha[:, i].view(B, 1, 1)
            updates = updates + a * XAB

        if squeezed:
            updates = updates.squeeze(1)

        return y_base + updates


def inject_mola_linear(
    model: torch.nn.Module,
    controller: ModeController,
    target_modules: List[str],
    num_experts: int,
    rank: int,
) -> torch.nn.Module:
    """Llama 내부 target Linear 모듈을 MoLA_Linear로 교체."""

    def _replace(module: torch.nn.Module):
        for name, child in list(module.named_children()):
            if name in target_modules and isinstance(child, torch.nn.Module) and hasattr(child, "weight"):
                mola = MoLA_Linear(child, controller, num_experts=num_experts, rank=rank)
                setattr(module, name, mola)
            else:
                _replace(child)

    _replace(model)
    return model


class ResidualMLP(torch.nn.Module):
    def __init__(self, dim, hidden, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden, dtype=dtype)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden, dim, dtype=dtype)
        self.norm = torch.nn.LayerNorm(dim, dtype=dtype)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return self.norm(x + h)


class SharedRouter(torch.nn.Module):
    """feature token 임베딩 평균 → α 계산."""

    def __init__(self, embed_dim: int, num_experts: int, hidden: int = ROUTER_HIDDEN_DIM, n_blocks: int = ROUTER_NUM_RESID, dtype=torch.bfloat16):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualMLP(embed_dim, hidden, dtype=dtype) for _ in range(n_blocks)])
        self.gate = torch.nn.Linear(embed_dim, num_experts, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        logits = self.gate(h)
        alpha = torch.nn.functional.softmax(logits, dim=-1)
        return alpha


class MoLAModel(torch.nn.Module):
    """
    학습 때 썼던 구조를 그대로 재구성하여 state_dict를 로드/추론에 사용.
    """

    def __init__(self, base_model: AutoModelForCausalLM, controller: ModeController, num_experts: int, feature_dim: int):
        super().__init__()
        self.base_model = base_model
        self.controller = controller

        self.shared_router = SharedRouter(feature_dim, num_experts)
        self.alpha_projector = torch.nn.Sequential(
            torch.nn.Linear(num_experts, ALPHA_PROJ_DIM, dtype=torch.bfloat16),
            torch.nn.GELU(),
            torch.nn.Linear(ALPHA_PROJ_DIM, ALPHA_PROJ_DIM, dtype=torch.bfloat16),
        )

    @torch.no_grad()
    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    # ✅ Train과 동일한 경로: feature_emb([B,D]) -> shared_router -> alpha
    def compute_alpha_from_feature_emb(self, feature_emb: torch.Tensor):
        """
        feature_emb: [B, D] (make_mola_data.py에서 정의된 pooled embedding)
        """
        alpha = self.shared_router(feature_emb.to(self.base_model.get_input_embeddings().weight.device))
        self.controller.shared_alpha = alpha
        return alpha

    def compute_alpha_from_features(self, feat_input_ids: torch.Tensor, feat_attention_mask: torch.Tensor):
        emb = self.base_model.get_input_embeddings()(feat_input_ids)
        pooled = self._masked_mean(emb, feat_attention_mask)
        alpha = self.shared_router(pooled)
        self.controller.shared_alpha = alpha
        return alpha

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def _move_mola_additions_to_device(model: MoLAModel, device: torch.device):
    """
    base_model은 accelerate가 배치한 디바이스를 그대로 두고,
    MoLA가 추가로 만든 파라미터만 지정된 디바이스로 옮긴다.
    """
    for module in model.base_model.modules():
        if isinstance(module, MoLA_Linear):
            for p in module.As:
                p.data = p.data.to(device)
            for p in module.Bs:
                p.data = p.data.to(device)
    model.shared_router.to(device)
    model.alpha_projector.to(device)




######################################
# Update the paths below for the target experiment before running.
# 첫 질문부터 상담가 유형별로 다르게 생성되도록 설정한 것

# --- 1. 사용자 설정 ---

# 1-1. 테스트할 상담가 버전 목록 (LoRA 어댑터 명시)
COUNSELOR_ADAPTER_ROOT = ""

DEFAULT_COUNSELOR_SYSTEM_PROMPT = """
You are a supportive and empathetic counseling assistant.
Understand the user's feelings, validate emotions, and respond in a calm, non-judgmental way.
Ask gentle clarifying questions only when helpful, and offer practical coping suggestions when appropriate.
Do not diagnose, prescribe, or claim professional authority. Keep responses concise and warm (3–6 sentences).
""".strip()

# base_counselor 전용 프롬프트(LoRA 없이 베이스만 사용할 때)
BASE_COUNSELOR_SYSTEM_PROMPT = """
You are a supportive and empathetic counseling assistant.
Understand the user's feelings, validate emotions, and respond in a calm, non-judgmental way.
Ask gentle clarifying questions only when helpful, and offer practical coping suggestions when appropriate.
Do not diagnose, prescribe, or claim professional authority.

Formatting constraints:
- Keep the response concise and warm.
- Use exactly 2–4 sentences.
- Ask at most one question.
- Avoid long explanations, lists, or repeated reassurance.
""".strip()

COUNSELOR_CONFIGURATIONS = [
    # {
    #     "counselor_name": "base_counselor_qwen",
    #     "adapter_path": None,  # LoRA 미적용
    #     "system_prompt": BASE_COUNSELOR_SYSTEM_PROMPT,
    # },
    {
        "counselor_name": "qwen-2.5-3b-multiwoz-finetuned",
        "adapter_path": None,  # LoRA 미적용
        "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    },
    # {
    #     "counselor_name": "counselchat_lr2e-05",
    #     "adapter_path": f"{COUNSELOR_ADAPTER_ROOT}/counselchat_lr2e-05__checkpoint-112/checkpoint-112",
    #     "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    # },
    # {
    #     "counselor_name": "esconv_lr2e-05",
    #     "adapter_path": f"{COUNSELOR_ADAPTER_ROOT}/esconv_lr2e-05__checkpoint-1106/checkpoint-1106",
    #     "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    # },
    # {
    #     "counselor_name": "extes_lr2e-05",
    #     "adapter_path": f"{COUNSELOR_ADAPTER_ROOT}/extes_lr2e-05__checkpoint-5867/checkpoint-5867",
    #     "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    # },
    # {
    #     "counselor_name": "psych8k_lr2e-05",
    #     "adapter_path": f"{COUNSELOR_ADAPTER_ROOT}/psych8k_lr2e-05__checkpoint-512/checkpoint-512",
    #     "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    # },
    # {
    #     "counselor_name": "psyinsight_lr2e-05",
    #     "adapter_path": f"{COUNSELOR_ADAPTER_ROOT}/psyinsight_lr2e-05/checkpoint-171",
    #     "system_prompt": DEFAULT_COUNSELOR_SYSTEM_PROMPT,
    # },
]


# 변경 확인하기
# [변경] 1-2. 테스트할 모델 경로를 **새로운 학습 결과물 경로**로 수정해야 합니다.
# 예시: base_output_dir = "./outputs/sft"

# 변경 확인하기
# [설명] 버전별 Instruction은 학습 방식과 관계없이 프롬프트의 일부이므로 그대로 둡니다.
CLIENT_CONFIGURATIONS = [
    {
        "version_name": "mola_ordinal_ds05_ne4_lr1e-3_rank4_twd5_tm_full_linear",
        "adapter_path": None,
        "mola_checkpoint_path": "",
        "mola_base_model_id": "",
        "test_set_path": ""
    },
]

# [설명] 버전별 Instruction은 학습 방식과 관계없이 프롬프트의 일부이므로 그대로 둡니다.
CLIENT_INSTRUCTION_BY_VERSION = {
    # 기존 st_... 버전에 대한 instruction은 여기에 그대로 두거나, 새로운 wo-st_... 버전에 맞게 수정/추가할 수 있습니다.
    "base_old": (
        "Based on the dialogue and your client features, generate your next utterance. If the conversation should end, output <|end_of_dialogue|> after your utterance."
    ),
}
# system prompt only에서는 None으로 설정해야 user 메시지 추가 안됨
DEFAULT_CLIENT_INSTRUCTION = None

NOISE_MARKERS = ["no_end_of_dialogue", "next sentence"]

# 1-3. 기본 모델 및 OpenAI 정보 (기존과 동일)
#BASE_MODEL_ID = ""
#BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_ID = "holi-lab/qwen-2.5-3b-multiwoz-finetuned"
COUNSELOR_BASE_MODEL = BASE_MODEL_ID
FIRST_UTT_MAP_PATH = ""

# 1-4. 대화 흐름 설정

# ================== [실험 파라미터] ==================
# 여기서 모든 실험 설정을 관리합니다.
MAX_TURNS = 10
PERSONA_INDICES_TO_TEST = list(range(0, 300))
# PERSONA_INDICES_TO_TEST = [] # 전체 다 돌릴 때
REPETITION_PENALTY = 1.1 
# 최소 대화 턴 수 필터 (None이면 전체 페르소나 사용)
MIN_TOTAL_DIALOGUE_TURNS = None
# ===================================================

# [자동 생성] 파일명에 사용할 문자열 생성
TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M")

# [수정] PERSONA_INDICES_TO_TEST 리스트가 비어있는 경우를 처리
if PERSONA_INDICES_TO_TEST: # 리스트에 내용이 있으면
    start_idx = PERSONA_INDICES_TO_TEST[0]
    end_idx = PERSONA_INDICES_TO_TEST[-1]
    idx_info_str = f"idx{start_idx}-{end_idx}"
else: # 리스트가 비어있으면 (전체 데이터)
    idx_info_str = "idx_all"

rep_pen_str = str(REPETITION_PENALTY).replace('.', 'p')

# [수정] idx_info_str 변수를 사용하여 로그 디렉토리 경로를 동적으로 생성
#LOG_BASE_DIR_TEMPLATE = f""
LOG_BASE_DIR_TEMPLATE = f""

GENERATION_SETTINGS = {
    "max_new_tokens": 350,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.9,
}


# --- 헬퍼 함수 ---
def require_config(name: str, value):
    if not value:
        raise ValueError(f"Please set `{name}` before running this script.")
    return value


def validate_client_config(client_config):
    version_name = client_config.get("version_name", "<unknown>")
    require_config(f"CLIENT_CONFIGURATIONS[{version_name}].test_set_path", client_config.get("test_set_path"))
    if client_config.get("mola_checkpoint_path"):
        require_config(
            f"CLIENT_CONFIGURATIONS[{version_name}].mola_base_model_id",
            client_config.get("mola_base_model_id") or BASE_MODEL_ID,
        )


def save_run_metadata(log_dir, counselor_config, client_config, persona_count, generation_kwargs):
    metadata_path = os.path.join(log_dir, "run_metadata.txt")
    counselor_summary = {"counselor_name": counselor_config.get("counselor_name")}
    all_counselor_summaries = [
        {"counselor_name": c.get("counselor_name")} for c in COUNSELOR_CONFIGURATIONS
    ]
    metadata = {
        "timestamp": TIMESTAMP,
        "log_directory": log_dir,
        "counselor_configuration": counselor_summary,
        "all_counselor_configurations": all_counselor_summaries,
        "client_configuration": client_config,
        "all_client_configurations": CLIENT_CONFIGURATIONS,
        "max_turns": MAX_TURNS,
        "persona_indices_to_test": PERSONA_INDICES_TO_TEST,
        "selected_persona_count": persona_count,
        "repetition_penalty": REPETITION_PENALTY,
        "min_total_dialogue_turns": MIN_TOTAL_DIALOGUE_TURNS,
        "generation_inference_mode": True,
        "generation_kwargs": generation_kwargs,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2))


def extract_total_dialogue_turns(persona_prompt):
    match = re.search(r'total_dialogue_turns:\s*(\d+)', persona_prompt)
    if match:
        return int(match.group(1))
    return None

def load_first_utterance_map(jsonl_path):
    """
    dialogue_id -> op_summary_first_person 맵 로드
    """
    mapping = {}
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            did = data.get("dialogue_id")
            if not did:
                continue
            utt = data.get("op_summary_first_person") or data.get("op_summary_first_person_raw") or data.get("op_summary")
            if not utt:
                continue
            mapping[str(did)] = utt.strip()
            cnt += 1
    print(f"Loaded {cnt} first-utterance mappings from {jsonl_path}")
    return mapping

def load_counselor_base_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(COUNSELOR_BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        COUNSELOR_BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    return base, tok


def attach_counselor_adapter(base_model, adapter_path):
    if not adapter_path:
        return base_model
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    ).eval()


def counselor_generate(counselor_model, counselor_tok, messages):
    input_ids = counselor_tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(counselor_model.device)

    with torch.inference_mode():
        out = counselor_model.generate(
            input_ids,
            max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
            do_sample=GENERATION_SETTINGS["do_sample"],
            temperature=GENERATION_SETTINGS["temperature"],
            top_p=GENERATION_SETTINGS["top_p"],
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=counselor_tok.pad_token_id,
        )

    gen = out[0][input_ids.shape[-1]:]
    return counselor_tok.decode(gen, skip_special_tokens=True).strip()



def extract_unique_personas(file_path, min_turns=None):
    print(f"Extracting unique personas from {file_path}...")
    unique_personas = []
    seen_prompts = set()
    filtered_out_count = 0
    unknown_turn_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['messages'] and data['messages'][0]['role'] == 'system':
                    system_prompt = data['messages'][0]['content']
                    dialogue_id = data.get("metadata", {}).get("dialogue_id")
                    if system_prompt in seen_prompts:
                        continue

                    total_turns = extract_total_dialogue_turns(system_prompt)
                    if min_turns is not None:
                        if total_turns is None:
                            unknown_turn_count += 1
                            continue
                        if total_turns < min_turns:
                            filtered_out_count += 1
                            continue

                    seen_prompts.add(system_prompt)
                    unique_personas.append({"system_prompt": system_prompt, "dialogue_id": dialogue_id, "feature_emb": data["feature_emb"]})
        print(f"Found {len(unique_personas)} unique personas.")
        if min_turns is not None:
            print(f"Filtered out {filtered_out_count} personas below {min_turns} turns.")
            if unknown_turn_count:
                print(f"Skipped {unknown_turn_count} personas with unknown turn counts.")
        return unique_personas
    except FileNotFoundError:
        print(f"Error: Test set file not found at {file_path}")
        return []


def load_sft_base_model():
    print("--- Loading Base Model and Tokenizer (SFT) ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    if hasattr(base_model.config, "cache_implementation"):
        base_model.config.cache_implementation = "static"
    if hasattr(base_model.config, "return_legacy_cache"):
        base_model.config.return_legacy_cache = True
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.use_cache = False
        if hasattr(base_model.generation_config, "cache_implementation"):
            base_model.generation_config.cache_implementation = "static"
        if hasattr(base_model.generation_config, "return_legacy_cache"):
            base_model.generation_config.return_legacy_cache = True
    base_model.eval()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    print("--- Base Model and Tokenizer Loaded Successfully ---")
    return base_model, tokenizer, terminators


def load_mola_checkpoint(
    checkpoint_path: str,
    base_model_id: str,
    feature_dim: int,
    num_experts: Optional[int] = None,
    rank: Optional[int] = None,
):
    """
    학습 때 저장한 MoLA state_dict(model.safetensors)을 로드해 추론용 모델을 구성.
    """
    print(f"--- Loading MoLA checkpoint from {checkpoint_path} ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    if hasattr(base_model.config, "cache_implementation"):
        base_model.config.cache_implementation = "static"
    if hasattr(base_model.config, "return_legacy_cache"):
        base_model.config.return_legacy_cache = True

    resolved_num_experts = num_experts if num_experts is not None else NUM_EXPERTS
    resolved_rank = rank if rank is not None else MOLA_RANK
    controller = ModeController()
    base_model = inject_mola_linear(
        base_model,
        controller,
        TARGET_MODULES,
        num_experts=resolved_num_experts,
        rank=resolved_rank,
    )

    model = MoLAModel(
        base_model,
        controller,
        num_experts=resolved_num_experts,
        feature_dim=feature_dim,
    )


    # 디버깅용 수정
    state_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"MoLA checkpoint not found at {state_path}")

    print(f"[DEBUG] Safetensors path: {state_path}")
    print(f"[DEBUG] File size: {os.path.getsize(state_path)} bytes")

    # --- 1단계: safetensors low-level로 실제 안에 뭐가 있는지 확인 ---
    try:
        with safe_open(state_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            meta = f.metadata()
        print(f"[DEBUG] safe_open OK. num_tensors={len(keys)}")
        print(f"[DEBUG] first 10 keys: {keys[:10]}")
        print(f"[DEBUG] metadata: {meta}")
    except Exception as e:
        print(f"[DEBUG] safe_open FAILED: {repr(e)}")
        # 여기서 터지면 파일 자체가 깨졌거나 safetensors 버전 이슈 가능성 ↑
        raise

    # --- 2단계: 일반 state_dict로 로드 (ModelWrapper 기대 X) ---
    try:
        print("[DEBUG] Trying safetensors.torch.load_file(...)")
        state_dict = load_file(state_path)
        print(f"[DEBUG] load_file OK. state_dict len={len(state_dict)}")
    except Exception as e:
        print(f"[DEBUG] load_file FAILED: {repr(e)}")
        raise

    # --- 3단계: MoLAModel 에 state_dict 주입 ---
    print("[DEBUG] Loading state_dict into MoLAModel (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[DEBUG] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[DEBUG] Missing keys example:", missing[:10])
    if unexpected:
        print("[DEBUG] Unexpected keys example:", unexpected[:10])




    # 새로 추가된 파라미터(MoLA 부분)만 임베딩 디바이스로 이동
    emb_device = base_model.get_input_embeddings().weight.device
    _move_mola_additions_to_device(model, emb_device)

    model.eval()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    print("--- MoLA Model Loaded Successfully ---")
    return model, tokenizer, terminators


def prepare_mola_alpha(model: MoLAModel, feature_emb: List[float]):
    device = model.base_model.get_input_embeddings().weight.device
    feature_tensor = torch.tensor(
        feature_emb,
        dtype=torch.bfloat16,
        device=device,
    ).unsqueeze(0)  # [1, D]

    model.compute_alpha_from_feature_emb(feature_tensor)



# None이 들어감
# def build_client_prompt(base_system_prompt, conversation_history, version_name=None):
#     messages = [{"role": "system", "content": base_system_prompt}]
#     for utt in conversation_history:
#         role = "user" if utt['speaker'] == 'counselor' else "assistant"
#         messages.append({"role": role, "content": utt['utterance']})
#     instruction = CLIENT_INSTRUCTION_BY_VERSION.get(version_name, DEFAULT_CLIENT_INSTRUCTION)
#     messages.append({"role": "user", "content": instruction})
#     return messages


def build_client_prompt(base_system_prompt, conversation_history, version_name=None):
    messages = [{"role": "system", "content": base_system_prompt}]

    # client에게는 데이터에서 제공된 첫 client 발화를 제외하고 넣어준다
    if conversation_history and conversation_history[0]["speaker"] == "client":
        history_for_client = conversation_history[1:]
    else:
        history_for_client = conversation_history

    for utt in history_for_client:
        role = "user" if utt['speaker'] == 'counselor' else "assistant"
        messages.append({"role": role, "content": utt['utterance']})

    instruction = CLIENT_INSTRUCTION_BY_VERSION.get(version_name, DEFAULT_CLIENT_INSTRUCTION)

    # 🔵 system prompt only 버전에서는 DEFAULT_CLIENT_INSTRUCTION = None 이니까
    #    이 if를 통과하지 않고, 추가 user 턴이 안 붙음 → 학습 포맷과 동일
    if instruction is not None:
        messages.append({"role": "user", "content": instruction})

    return messages


# [변경] === Client generation 함수 단순화 ===
# [설명] EOD를 일반 텍스트로 학습했으므로, 더 이상 스페셜 토큰 ID를 추적하거나 복잡한 확률 계산을 할 필요가 없습니다.
#       단순히 텍스트를 생성하고, 그 안에 EOD 문자열이 포함되어 있는지 여부만 확인하면 됩니다.
# --- [개선 2] Client generation 함수 안정성 강화 ---
def get_client_response(model, tokenizer, client_prompt_messages, terminators, use_cache=True):
    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        client_prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        generation_kwargs = dict(GENERATION_SETTINGS)
        # generation_kwargs.setdefault("cache_implementation", "static")
        # generation_kwargs.setdefault("return_legacy_cache", True)
        if not use_cache:
            generation_kwargs["use_cache"] = False
            generation_kwargs["cache_implementation"] = "static"
            generation_kwargs["return_legacy_cache"] = True
        out = model.generate(
            input_ids,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators, 
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
            **generation_kwargs,
        )

    generated_tokens = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
    prob_info = {}
    return text, prob_info

# --- [개선 1] 메인 실행 로직 효율성 강화 ---

# run_simulation 함수는 이제 로드된 모델과 토크나이저를 인자로 받습니다.
def run_simulation(base_model, client_tokenizer, client_config, counselor_config, personas, terminators, counselor_model, counselor_tokenizer, first_utt_map):
    client_version_name = client_config['version_name']
    adapter_path = client_config.get('adapter_path')
    mola_checkpoint_path = client_config.get("mola_checkpoint_path")
    counselor_name = counselor_config['counselor_name']
    counselor_system_prompt = counselor_config['system_prompt']

    is_mola = bool(mola_checkpoint_path)
    using_adapter = bool(adapter_path) and (not is_mola)
    resolved_num_experts = client_config.get("_resolved_num_experts", NUM_EXPERTS)
    if is_mola:
        print(f"--- Using MoLA checkpoint for version: {client_version_name} ---")
        print(f"[INFO] NUM_EXPERTS for {client_version_name}: {resolved_num_experts}")
        client_model = base_model  # 이미 MoLA 구조가 주입된 모델
        client_model.eval()
    elif using_adapter:
        print(f"--- Attaching Adapter: {client_version_name} ---")
        # 전달받은 base_model에 어댑터만 교체하며 사용합니다.
        peft_config = PeftConfig.from_pretrained(adapter_path)
        if client_version_name == "prefix_tuning" and getattr(peft_config, "peft_type", "").upper() == "PREFIX_TUNING":
            setattr(peft_config, "modules_to_save", None)
        client_model = PeftModel.from_pretrained(base_model, adapter_path, config=peft_config)
        if hasattr(client_model, "config"):
            client_model.config.use_cache = False
            setattr(client_model.config, "cache_implementation", "static")
            setattr(client_model.config, "return_legacy_cache", True)
        if hasattr(client_model, "generation_config"):
            client_model.generation_config.use_cache = False
            setattr(client_model.generation_config, "cache_implementation", "static")
            setattr(client_model.generation_config, "return_legacy_cache", True)
        if hasattr(client_model, "base_model") and hasattr(client_model.base_model, "config"):
            client_model.base_model.config.use_cache = False
            setattr(client_model.base_model.config, "cache_implementation", "static")
            setattr(client_model.base_model.config, "return_legacy_cache", True)
        client_model.eval()
        print("Adapter attached successfully.")
    else:
        print(f"--- Using Base Model (no adapter) for version: {client_version_name} ---")
        client_model = base_model
        client_model.eval()
    
    log_base_dir = LOG_BASE_DIR_TEMPLATE.format(version_name=client_version_name)
    log_dir = os.path.join(log_base_dir, counselor_name, client_version_name)
    os.makedirs(log_dir, exist_ok=True)
    with_alpha_dir = os.path.join(log_dir, "with_alpha")
    os.makedirs(with_alpha_dir, exist_ok=True)

    persona_count = len(personas)
    terminator_ids = [int(t) if t is not None else None for t in terminators]
    use_cache_for_generation = not (client_version_name == "prefix_tuning")
    generation_kwargs_for_log = {
        **GENERATION_SETTINGS,
        "pad_token_id": client_tokenizer.pad_token_id,
        "eos_token_id": terminator_ids,
        "repetition_penalty": REPETITION_PENALTY,
        "use_cache": use_cache_for_generation,
    }
    save_run_metadata(log_dir, counselor_config, client_config, persona_count, generation_kwargs_for_log)

    # 🔵 여기 추가: 첫 번째 모델 입력만 한 번 저장하기 위한 플래그
    saved_first_input_prompt = False

    import tqdm
    for idx, persona in tqdm.tqdm(enumerate(personas)):
        # ... (기존 for문 로직과 동일) ...
        # (단, get_client_response 호출 시 terminators를 넘겨줍니다)
        persona_id = idx + 1
        persona_prompt = persona["system_prompt"]
        dialogue_id = persona.get("dialogue_id")
        safe_dialogue_id = None
        if dialogue_id:
            safe_dialogue_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", dialogue_id)
        log_filename = f"conversation_log_persona_{persona_id}"
        if safe_dialogue_id:
            log_filename += f"_{safe_dialogue_id}"
        log_filename += ".jsonl"
        log_file_path = os.path.join(log_dir, log_filename)
        print(f"\n--- Running Persona {persona_id}/{len(personas)} for ({counselor_name} vs {client_version_name}) ---")
        
        with_alpha_log_path = os.path.join(with_alpha_dir, log_filename)


        dialogue_id = persona.get("dialogue_id")
        if not dialogue_id or str(dialogue_id) not in first_utt_map:
            print(f"[Persona {persona_id}] dialogue_id '{dialogue_id}' has no first utterance mapping. Skipping.")
            continue

        first_utt = first_utt_map[str(dialogue_id)]

        conversation_log = []
        history_for_counselor = [
            {"role": "system", "content": counselor_system_prompt},
            {"role": "user", "content": first_utt},
        ]

        # 첫 페르소나 입력 프롬프트를 턴별로 저장
        save_inputs = (idx == 0)
        counselor_input_f = client_input_f = None
        if save_inputs:
            base_fn = f"persona_{persona_id}"
            if safe_dialogue_id:
                base_fn += f"_{safe_dialogue_id}"
            counselor_input_path = os.path.join(log_dir, f"counselor_inputs_{base_fn}.txt")
            client_input_path = os.path.join(log_dir, f"client_inputs_{base_fn}.txt")
            counselor_input_f = open(counselor_input_path, "w", encoding="utf-8")
            client_input_f = open(client_input_path, "w", encoding="utf-8")

        with open(log_file_path, 'w', encoding='utf-8') as f, \
            open(with_alpha_log_path, 'w', encoding='utf-8') as f_alpha:

            f.write(json.dumps({"persona_id": persona_id, "persona_prompt": persona_prompt, "dialogue_id": dialogue_id}, ensure_ascii=False) + '\n')
            f_alpha.write(json.dumps({"persona_id": persona_id, "persona_prompt": persona_prompt, "dialogue_id": dialogue_id}, ensure_ascii=False) + '\n')
            log_item = {"speaker": "client", "utterance": first_utt}
            print(f"[Turn 0] Client: {first_utt}")
            f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
            f_alpha.write(json.dumps(log_item, ensure_ascii=False) + '\n')
            conversation_log.append(log_item)

            for i in range(MAX_TURNS):
                turn = i + 1

                if counselor_input_f is not None:
                    counselor_input_text = counselor_tokenizer.apply_chat_template(
                        history_for_counselor,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    counselor_input_f.write(f"===== TURN {turn} =====\n")
                    counselor_input_f.write(counselor_input_text + "\n\n")

                counselor_utterance = counselor_generate(
                    counselor_model,
                    counselor_tokenizer,
                    history_for_counselor
                )
                log_item = {"speaker": "counselor", "utterance": counselor_utterance}
                print(f"[Turn {turn}] Counselor: {counselor_utterance}")
                f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                f_alpha.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                conversation_log.append(log_item)
                history_for_counselor.append({"role": "assistant", "content": counselor_utterance})

                client_prompt = build_client_prompt(persona_prompt, conversation_log, client_version_name)

                if client_input_f is not None:
                    client_input_text = client_tokenizer.apply_chat_template(
                        client_prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    client_input_f.write(f"===== TURN {turn} =====\n")
                    client_input_f.write(client_input_text + "\n\n")

                                # MoLA는 turn마다 shared_alpha가 필요하므로 persona별 feature로 한 번 계산

                if is_mola and i == 0:
                    prepare_mola_alpha(client_model, persona["feature_emb"])

                    alpha = client_model.controller.shared_alpha.detach().cpu().float().squeeze(0)
                    alpha_list = alpha.tolist()
                    dominant_expert = int(alpha.argmax().item())
                    alpha_entropy = float(-(alpha * torch.log(alpha + 1e-12)).sum().item())
                    alpha_header = {
                        "persona_id": persona_id,
                        "dialogue_id": dialogue_id,
                        "alpha": alpha_list,
                        "dominant_expert": dominant_expert,
                        "alpha_entropy": alpha_entropy,
                    }

                    f_alpha.write(json.dumps(alpha_header, ensure_ascii=False) + "\n")



                # 🔵 [추가] 맨 처음 한 번만 모델 입력 프롬프트 저장
                if (not saved_first_input_prompt) and idx == 0 and i == 0:
                    debug_prompt_text = client_tokenizer.apply_chat_template(
                        client_prompt,
                        add_generation_prompt=True,  # generate와 동일하게
                        tokenize=False
                    )
                    debug_path = os.path.join(log_dir, "first_model_input_prompt.txt")
                    with open(debug_path, "w", encoding="utf-8") as df:
                        df.write(debug_prompt_text)
                    print(f"[DEBUG] Saved first model input prompt to: {debug_path}")
                    saved_first_input_prompt = True
                
                # terminators를 인자로 전달
                client_utterance_raw, prob_info = get_client_response(
                    client_model,
                    client_tokenizer,
                    client_prompt,
                    terminators,
                    use_cache=use_cache_for_generation
                )

                eod_string = "<|end_of_dialogue|>"
                client_utterance = client_utterance_raw

                if client_version_name == "base":
                    detection_pattern = re.compile(r'(?<![A-Za-z0-9_])end_of_dialogue(?![A-Za-z0-9_])', re.IGNORECASE)
                    match = detection_pattern.search(client_utterance_raw)
                    eod_found = bool(match)

                    if match:
                        start = match.start()
                        while start > 0 and client_utterance_raw[start - 1] in " \t\n\r|<>()[]{}":
                            start -= 1
                        end = match.end()
                        while end < len(client_utterance_raw) and client_utterance_raw[end] in " \t\n\r|<>()[]{}":
                            end += 1
                        client_utterance = client_utterance_raw[:start].rstrip()

                    note_match = re.search(r'(?i)(?:^|[\s\(])note:\s*', client_utterance)
                    if note_match:
                        client_utterance = client_utterance[:note_match.start()].rstrip()
                    if "\n\n" in client_utterance:
                        before, after = client_utterance.split("\n\n", 1)
                        lowered_after = after.lower()
                        if any(marker in lowered_after for marker in NOISE_MARKERS):
                            client_utterance = before.rstrip()
                    for marker in NOISE_MARKERS:
                        pattern = re.compile(
                            r'[\{\[\(\s]*\|?' + re.escape(marker) + r'\|?[\}\]\)\s]*',
                            re.IGNORECASE
                        )
                        client_utterance = pattern.sub('', client_utterance)
                    client_utterance = client_utterance.rstrip()
                    if len(client_utterance) >= 2 and client_utterance[0] == client_utterance[-1] == '"':
                        client_utterance = client_utterance[1:-1].strip()
                elif client_version_name == "wo_features":
                    eod_found = eod_string in client_utterance_raw
                    if eod_found:
                        client_utterance = client_utterance_raw.split(eod_string)[0].rstrip()
                    note_match = re.search(r'(?i)(?:^|[\s\(])note:\s*', client_utterance)
                    if note_match:
                        client_utterance = client_utterance[:note_match.start()].rstrip()
                    if "\n\n" in client_utterance:
                        before, after = client_utterance.split("\n\n", 1)
                        lowered_after = after.lower()
                        if any(marker in lowered_after for marker in NOISE_MARKERS):
                            client_utterance = before.rstrip()
                    for marker in NOISE_MARKERS:
                        pattern = re.compile(
                            r'[\{\[\(\s]*\|?' + re.escape(marker) + r'\|?[\}\]\)\s]*',
                            re.IGNORECASE
                        )
                        client_utterance = pattern.sub('', client_utterance)
                    client_utterance = client_utterance.rstrip()
                    if len(client_utterance) >= 2 and client_utterance[0] == client_utterance[-1] == '"':
                        client_utterance = client_utterance[1:-1].strip()
                else:
                    eod_found = eod_string in client_utterance_raw
                    if eod_found:
                        client_utterance = client_utterance_raw.split(eod_string)[0].rstrip()


                log_item = {
                    "speaker": "client",
                    "utterance": client_utterance,
                    "eod_generated": eod_found, 
                    "raw_output": client_utterance_raw,
                    **prob_info
                }
                
                print(f"[Turn {turn}] Client: {client_utterance}")
                f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                f_alpha.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                conversation_log.append({"speaker": "client", "utterance": client_utterance})
                history_for_counselor.append({"role": "user", "content": client_utterance})

                if eod_found:
                    print("--- EOD Detected. Ending conversation. ---")
                    break

            if counselor_input_f is not None:
                counselor_input_f.close()
            if client_input_f is not None:
                client_input_f.close()

    print(f"--- All persona simulations for ({counselor_name} vs {client_version_name}) completed. ---")
    
    # 🔵 변경
    if using_adapter:
        del client_model
    gc.collect()
    torch.cuda.empty_cache()



def main():
    #args = parse_args()
    if not CLIENT_CONFIGURATIONS or not COUNSELOR_CONFIGURATIONS:
        print("Configurations are missing. Exiting.")
        return
    require_config("LOG_BASE_DIR_TEMPLATE", LOG_BASE_DIR_TEMPLATE)
    require_config("FIRST_UTT_MAP_PATH", FIRST_UTT_MAP_PATH)
    for client_config in CLIENT_CONFIGURATIONS:
        validate_client_config(client_config)

    base_model_cache = None
    tokenizer_cache = None
    terminators_cache = None
    completed_log_dirs = set()

    first_utt_map = load_first_utterance_map(FIRST_UTT_MAP_PATH)

    for counselor_config in COUNSELOR_CONFIGURATIONS:
        counselor_name = counselor_config['counselor_name']
        adapter_path = counselor_config.get("adapter_path")
        print(f"\n{'#'*40}\n# Starting All Tests for Counselor Version: {counselor_name}\n{'#'*40}")

        print("--- Loading Counselor Base/Tokenizer ---")
        counselor_base, counselor_tokenizer = load_counselor_base_and_tokenizer()
        counselor_model = attach_counselor_adapter(counselor_base, adapter_path)
        counselor_model.eval()
        print("--- Counselor Loaded ---")

        for client_config in CLIENT_CONFIGURATIONS:
            client_config = dict(client_config)
            is_mola = bool(client_config.get("mola_checkpoint_path"))
            client_model = None
            #try:
            min_turns = None if client_config.get('version_name') == "wo_features" else MIN_TOTAL_DIALOGUE_TURNS
            personas = extract_unique_personas(
                client_config['test_set_path'],
                min_turns
            )
            if not personas:
                print(f"No personas found for client {client_config['version_name']}. Skipping.")
                continue
            
            # if NUM_PERSONAS_TO_TEST is not None and NUM_PERSONAS_TO_TEST > 0:
            #     personas = personas[:min(NUM_PERSONAS_TO_TEST, len(personas))]

            if PERSONA_INDICES_TO_TEST: # 리스트가 비어있지 않은 경우에만 실행
                try:
                    # 지정된 인덱스에 해당하는 페르소나만 선택
                    personas = [personas[i] for i in PERSONA_INDICES_TO_TEST]
                    print(f"Selected {len(personas)} specific personas based on indices: {PERSONA_INDICES_TO_TEST}")
                except IndexError:
                    print(f"\n[Warning] An index in {PERSONA_INDICES_TO_TEST} is out of range for {len(personas)} available personas. Please check your indices.")
                    print("Skipping this combination.")
                    continue

            # 조합별 모델/토크나이저 준비
            if is_mola:
                mola_base_id = client_config.get("mola_base_model_id", BASE_MODEL_ID)
                # Determine NUM_EXPERTS for this checkpoint (wandb_config override or default).
                resolved_num_experts = resolve_num_experts_from_checkpoint(
                    client_config["mola_checkpoint_path"],
                    fallback=NUM_EXPERTS,
                )
                resolved_mola_rank = resolve_mola_rank_from_checkpoint(
                    client_config["mola_checkpoint_path"],
                    fallback=MOLA_RANK,
                )
                client_config["_resolved_num_experts"] = resolved_num_experts
                client_config["_resolved_mola_rank"] = resolved_mola_rank
                print(
                    f"[INFO] Using NUM_EXPERTS={resolved_num_experts}, "
                    f"MOLA_RANK={resolved_mola_rank} for {client_config['mola_checkpoint_path']}"
                )
                feature_dim = infer_feature_dim_from_testset(
                    client_config["test_set_path"]
                )
                client_model, client_tokenizer, terminators = load_mola_checkpoint(
                    client_config["mola_checkpoint_path"],
                    mola_base_id,
                    feature_dim=feature_dim,
                    num_experts=resolved_num_experts,
                    rank=resolved_mola_rank,
                )
            else:
                if base_model_cache is None:
                    base_model_cache, tokenizer_cache, terminators_cache = load_sft_base_model()
                client_model = base_model_cache
                client_tokenizer = tokenizer_cache
                terminators = terminators_cache

            run_simulation(client_model, client_tokenizer, client_config, counselor_config, personas, terminators, counselor_model, counselor_tokenizer, first_utt_map)
            log_base_dir = LOG_BASE_DIR_TEMPLATE.format(version_name=client_config['version_name'])
            completed_log_dirs.add(log_base_dir)

            # except Exception as e:
            #     print(f"\nAn error occurred while processing ({counselor_name} vs {client_config.get('version_name')}): {e}")
            #     print("Skipping to the next combination.")
            # finally:
            #     if is_mola and client_model is not None:
            #         del client_model
            #         gc.collect()
            #         torch.cuda.empty_cache()

        del counselor_model
        del counselor_base
        torch.cuda.empty_cache()

    if completed_log_dirs:
        dir_lines = "\n".join(f"- {path}" for path in sorted(completed_log_dirs))
        print(f"\n{'='*40}\n# All Simulations Completed. Logs are saved under:\n{dir_lines}\n{'='*40}")
    else:
        base_dir_hint = LOG_BASE_DIR_TEMPLATE.replace("{version_name}", "<version_name>")
        print(f"\n{'='*40}\n# All Simulations Completed. Logs are saved under '{base_dir_hint}'\n{'='*40}")

if __name__ == "__main__":
    main()
