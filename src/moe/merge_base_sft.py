#!/usr/bin/env python
# Merge base model and LoRA adapter into a single HF folder.
# - base, adapter_root는 아래 상수를 수정해서 사용.
# - adapter_root에 checkpoint-*가 있으면 가장 큰 스텝을 자동 선택.
# - Set RESULTS_DIR to choose where merged models are saved.

import os
import re
import json
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ====================== 경로 설정 ======================
BASE_MODEL_PATH = ""
ADAPTER_ROOT = ""
RESULTS_DIR = ""

# dtype/device 설정 (필요에 따라 수정)
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16
DEVICE_MAP = "auto"  # 여러 GPU면 자동 배치, 단일 GPU면 알아서 하나만 사용


def require_config(name: str, value):
    if not value:
        raise ValueError(f"Please set `{name}` before running this script.")
    return value


def find_latest_checkpoint(adapter_root: str) -> str:
    """adapter_root 내에서 가장 큰 스텝의 checkpoint-* 폴더를 반환."""
    pattern = re.compile(r"checkpoint-(\d+)")
    best: Optional[tuple[int, str]] = None

    for name in os.listdir(adapter_root):
        match = pattern.fullmatch(name)
        if not match:
            continue
        step = int(match.group(1))
        full = os.path.join(adapter_root, name)
        if not os.path.isdir(full):
            continue
        if best is None or step > best[0]:
            best = (step, full)

    return best[1] if best else adapter_root


def main():
    require_config("BASE_MODEL_PATH", BASE_MODEL_PATH)
    require_config("ADAPTER_ROOT", ADAPTER_ROOT)
    require_config("RESULTS_DIR", RESULTS_DIR)

    adapter_path = find_latest_checkpoint(ADAPTER_ROOT)
    adapter_dir_name = os.path.basename(ADAPTER_ROOT.rstrip("/"))

    if adapter_path == ADAPTER_ROOT:
        output_dir = os.path.join(RESULTS_DIR, f"{adapter_dir_name}-merged")
        chosen_ckpt = "root (no checkpoint-*)"
    else:
        ckpt_name = os.path.basename(adapter_path)
        output_dir = os.path.join(RESULTS_DIR, f"{adapter_dir_name}-{ckpt_name}-merged")
        chosen_ckpt = ckpt_name

    os.makedirs(output_dir, exist_ok=True)

    # 토크나이저는 어댑터 폴더에서 로드 후 pad 토큰을 명시적으로 분리(add)해 vocab 증가
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ROOT)
    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = None  # pad를 EOS와 분리
    if tokenizer.pad_token is None:
        if "[PAD]" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["[PAD]"])
        tokenizer.pad_token = "[PAD]"
    vocab_size = len(tokenizer)

    print(f"[merge] Base:   {BASE_MODEL_PATH}")
    print(f"[merge] Adapter:{adapter_path} (chosen: {chosen_ckpt})")
    print(f"[merge] Output: {output_dir}")
    print(f"[merge] dtype={TORCH_DTYPE}, device_map={DEVICE_MAP}")
    print(f"[merge] vocab size after pad fix: {vocab_size}")
    print(f"[merge] pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    print("\nLoading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
    )

    # 어댑터가 사용한 vocab 크기에 맞게 임베딩 리사이즈
    base_model.resize_token_embeddings(vocab_size)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading adapter …")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )

    print("Merging LoRA into base weights …")
    merged_model = peft_model.merge_and_unload()

    print("Saving merged model …")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # merge metadata
    info = {
        "base_model": BASE_MODEL_PATH,
        "adapter_root": ADAPTER_ROOT,
        "adapter_used": adapter_path,
        "chosen_checkpoint": chosen_ckpt,
        "output_dir": output_dir,
        "torch_dtype": str(TORCH_DTYPE),
        "device_map": DEVICE_MAP,
    }
    with open(os.path.join(output_dir, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
