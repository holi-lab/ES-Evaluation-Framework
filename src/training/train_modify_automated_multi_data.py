import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import json # jsonl 파일 처리를 위해 필요
import wandb # wandb 임포트
from transformers import EarlyStoppingCallback # 수정
import gc # 가비지 컬렉션
from datetime import datetime 
from trl import DataCollatorForCompletionOnlyLM # 수정(추가)

# 하나의 lr에 대해 여러 데이터 경로 돌고
# 그 다음 lr로 이동

# wandb 로그인 (API 키가 설정되어 있지 않다면, 이 코드가 실행될 때 로그인 프롬프트가 나타납니다)
# 혹은 `wandb login` 명령어를 터미널에서 미리 실행할 수도 있습니다.
wandb.login()

# --- 기본 설정 ---
# Llama 3 토크나이저 및 모델 ID
model_id = ""
wandb_project = ""

# 학습에 사용할 데이터셋 경로 목록 (각 경로 안에 train*.jsonl, val*.jsonl 존재)
dataset_paths = []
# 1204_sft_muti_data2에서 맨 마지막거만, 1204_sft_muti_data에서는 3개 다 포함해서 

# # 실험할 학습률 목록 (필요에 맞게 수정)
# learning_rates = [5e-5, 8e-5, 2e-5] -> 앞 1개는 sft1, 뒤 2개는 sft2 tmux에서
learning_rates = [2e-5]

# 이름변경
# 학습 결과를 저장할 기본 출력 디렉토리
# 각 학습마다 이 디렉토리 아래에 '버전명'으로 하위 폴더가 생성됩니다.
# base_output_dir = "./outputs/sft"
base_output_dir = ""

if base_output_dir:
    os.makedirs(base_output_dir, exist_ok=True)


def require_config(name: str, value):
    if not value:
        raise ValueError(f"Please set `{name}` before running this script.")
    return value

# BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# LoraConfig 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,# 보통은 r*2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 학습률 문자열 포맷터
def format_learning_rate(value: float) -> str:
    """학습률을 경로/이름에서 읽기 쉬운 형태로 변환합니다."""
    formatted = f"{value:.0e}"
    return formatted.replace("e-0", "e-").replace("e+0", "e")


# 데이터셋 로드 함수
def load_jsonl_file(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def find_split_file(dataset_dir: str, split_prefix: str) -> str:
    """주어진 디렉토리에서 split_prefix로 시작하는 첫 번째 *.jsonl 파일을 찾습니다."""
    candidates = sorted(
        [
            f for f in os.listdir(dataset_dir)
            if f.startswith(split_prefix) and f.endswith(".jsonl")
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"No {split_prefix}*.jsonl found in {dataset_dir}")
    return os.path.join(dataset_dir, candidates[0])

# 데이터 포맷팅 함수 (오류 수정)
def formatting_prompts_func(examples, tokenizer_instance):
    output_texts = []
    for messages in examples["messages"]:
        if messages and messages[-1]["role"] == "assistant":
            # apply_chat_template이 Llama 3 형식에 맞게 EOS 토큰(<|eot_id|>)을 자동으로 추가합니다.
            # 따라서 수동으로 추가할 필요가 없습니다.
            text = tokenizer_instance.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False, # 마지막이 assistant 응답이므로 False
            )
            output_texts.append(text)
        else:
            # 학습 데이터로 사용하지 않을 샘플은 필터링을 위해 빈 문자열로 처리
            output_texts.append("")
    return {"text": output_texts}


# 토크나이즈 함수
def tokenize_function(examples, tokenizer_instance):
    return tokenizer_instance(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=2048, # 이번거 변경 # 시퀀스 길이, 1120 4096 으로 돌렸는데 아래로 다시 돌리기
        # max_length=2504
    )

# --- 학습 자동화를 위한 함수 ---
def train_model_for_version(version_name, dataset_dir, learning_rate):
    """특정 데이터 버전에 대해 모델 학습을 실행합니다."""
    print(f"\n--- Starting training for version: {version_name} ---")

    lr_tag = format_learning_rate(learning_rate)
    print(f"Learning rate: {learning_rate} (tag: {lr_tag})")

    train_file_path = find_split_file(dataset_dir, "train")
    val_file_path = find_split_file(dataset_dir, "val")
    
    # 학습 결과 저장될 특정 버전의 출력 디렉토리
    current_output_dir = os.path.join(base_output_dir, f"{version_name}_lr{lr_tag}")
    os.makedirs(current_output_dir, exist_ok=True)

    # 이름변경
    # Wandb 런 이름 설정
    wandb_run_name = f"llama-sft-all-data-{version_name}-lr{lr_tag}"
    
    # 이름변경
    # Wandb 런 초기화
    wandb.init(project=wandb_project, name=wandb_run_name)

    # 토크나이저 로드 (학습마다 새로 로드하여 상태 초기화)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # [수정]
    # eos도 -100으로 바꾸는 문제 해결 방법
    # 별도의 PAD 토큰 추가, 이를 공식 pad_token으로 지정
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"


    # 데이터 로드 및 포맷팅, 토크나이징
    train_raw_dataset = load_jsonl_file(train_file_path)
    eval_raw_dataset = load_jsonl_file(val_file_path)

    # [확인용]
    print("\n--- [VERIFICATION] Checking Tokenizer and Data Sample ---")

    # 1. 토크나이저 설정 정보 저장
    token_info_path = os.path.join(current_output_dir, "token_info.txt")
    with open(token_info_path, 'w', encoding='utf-8') as f:
        f.write("--- Tokenizer Settings Verification ---\n")
        f.write(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})\n")
        f.write(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})\n")
        f.write(f"Vocabulary size: {len(tokenizer)}\n")
        f.write("\n" + "="*50 + "\n\n")

        # 2. 실제 학습 데이터 첫 번째 샘플로 토큰화 결과 저장
        f.write("--- Sample Data Tokenization Verification ---\n")

        # 학습 데이터셋에서 첫 번째 샘플 가져오기
        sample_data = train_raw_dataset[0]

        # 템플릿 적용하여 텍스트로 변환
        sample_text = tokenizer.apply_chat_template(
            sample_data["messages"], tokenize=False, add_generation_prompt=False
        )
        f.write(f"Original Text (after template):\n---\n{sample_text}\n---\n\n")

        # 토큰화하여 input_ids 생성
        tokenized_sample = tokenizer(sample_text, return_tensors="pt")
        input_ids = tokenized_sample.input_ids[0]
        f.write(f"Input IDs:\n{input_ids.tolist()}\n\n")

        # 데이터 콜레이터를 통해 labels 생성
        temp_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_template,
        )
        collator_output = temp_collator(
            [{"input_ids": input_ids, "attention_mask": tokenized_sample.attention_mask[0]}]
        )
        labels = collator_output["labels"][0]
        f.write(f"Labels (after completion-only collator):\n{labels.tolist()}\n")

    print(f"Detailed token info with a sample saved to: {token_info_path}")
    print("----------------------------------------------------------\n")


    processed_train_dataset = train_raw_dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
        remove_columns=train_raw_dataset.column_names
    ).filter(lambda x: x["text"] != "")

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 모델 입력 파일 저장 코드 (첫 5개 샘플만) ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    
    save_path = os.path.join(current_output_dir, f"{version_name}_lr{lr_tag}_processed_inputs_5_samples.txt")
    print(f"\n--- Saving the first 5 processed model inputs for '{version_name}' to: {save_path} ---")

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            # processed_train_dataset의 모든 항목을 순회
            for i, item in enumerate(processed_train_dataset):
                # 5개 샘플만
                if i >= 5:
                    break
                
                f.write(f"==================== SAMPLE {i+1} ====================\n\n")
                f.write(item['text'])
                f.write("\n\n")

        print(f"--- Successfully saved the first 5 inputs. ---\n")
    
    except Exception as e:
        print(f"--- Error saving inputs to file: {e} ---\n")

    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 모델 입력 파일 저장 코드 종료 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # eos 토큰 포함 여부 확인용 출력
    print("\n--- Sample processed train data (last 50 chars) ---")
    for i in range(min(5, len(processed_train_dataset))):
        sample_text = processed_train_dataset[i]["text"]
        print(f"Sample {i+1} ends with: {sample_text[-50:]}")

    processed_eval_dataset = eval_raw_dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer), 
        batched=True, 
        remove_columns=eval_raw_dataset.column_names
    ).filter(lambda x: x["text"] != "")

    tokenized_train_dataset = processed_train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True
    )
    tokenized_eval_dataset = processed_eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True
    )

    print(f"Processed training dataset size for {version_name}: {len(processed_train_dataset)}")
    print(f"Processed validation dataset size for {version_name}: {len(processed_eval_dataset)}")

    # 샘플 데이터 확인
    print(f"\n--- Sample Processed Train Data for {version_name} (first 3 examples) ---")
    for i in range(min(3, len(processed_train_dataset))):
        print(f"Example {i+1}:\n{processed_train_dataset[i]['text'][:1000]}...\n") 

    # 모델 로드 (학습마다 새로 로드하여 상태 초기화)
    if 'model' in locals() and model is not None:
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # [수정]
    # 모델의 토큰 임베딩 크기를 새로운 토크나이저 크기에 맞게 조절
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # TrainingArguments 설정
    training_arguments = TrainingArguments(
        output_dir=current_output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        group_by_length=True,
        report_to="wandb",
        run_name=wandb_run_name,
        push_to_hub=False,
        load_best_model_at_end=True, # 베스트 모델을 로드하기 위한 설정 활성화
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 러닝 파라미터 및 데이터 요약 저장 (다른 txt 저장 시점에 함께 저장)
    run_details_path = os.path.join(current_output_dir, f"{version_name}_lr{lr_tag}_run_details.txt")
    run_details = {
        "timestamp": datetime.now().isoformat(),
        "version_name": version_name,
        "learning_rate": learning_rate,
        "lr_tag": lr_tag,
        "model_id": model_id,
        "dataset_dir": dataset_dir,
        "train_file": train_file_path,
        "val_file": val_file_path,
        "processed_train_samples": len(processed_train_dataset),
        "processed_eval_samples": len(processed_eval_dataset),
        "tokenizer": {
            "padding_side": tokenizer.padding_side,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        },
        "files": {
            "token_info": token_info_path,
            "processed_inputs_preview": save_path,
            "output_dir": current_output_dir,
        },
        "bnb_config": {
            "load_in_4bit": bnb_config.load_in_4bit,
            "bnb_4bit_quant_type": str(bnb_config.bnb_4bit_quant_type),
            "bnb_4bit_compute_dtype": str(bnb_config.bnb_4bit_compute_dtype),
            "bnb_4bit_use_double_quant": bnb_config.bnb_4bit_use_double_quant,
        },
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules) if isinstance(lora_config.target_modules, (list, tuple, set)) else lora_config.target_modules,
            "bias": lora_config.bias,
            "lora_dropout": lora_config.lora_dropout,
            "task_type": str(lora_config.task_type),
        },
        "training_arguments": {
            "per_device_train_batch_size": training_arguments.per_device_train_batch_size,
            "gradient_accumulation_steps": training_arguments.gradient_accumulation_steps,
            "optim": training_arguments.optim,
            "learning_rate": training_arguments.learning_rate,
            "lr_scheduler_type": str(training_arguments.lr_scheduler_type),
            "fp16": training_arguments.fp16,
            "bf16": training_arguments.bf16,
            "num_train_epochs": training_arguments.num_train_epochs,
            "logging_steps": training_arguments.logging_steps,
            "save_steps": training_arguments.save_steps,
            "eval_steps": training_arguments.eval_steps,
            "eval_strategy": str(training_arguments.eval_strategy),
            "group_by_length": training_arguments.group_by_length,
            "report_to": list(training_arguments.report_to) if isinstance(training_arguments.report_to, (list, tuple, set)) else training_arguments.report_to,
            "run_name": training_arguments.run_name,
            "load_best_model_at_end": training_arguments.load_best_model_at_end,
            "metric_for_best_model": training_arguments.metric_for_best_model,
            "greater_is_better": training_arguments.greater_is_better,
        },
        "wandb": {
            "project": wandb_project,
            "run_name": wandb_run_name,
        },
        "notes": "First 5 processed inputs saved to 'processed_inputs_preview'. Token verification in 'token_info'.",
    }
    # json.dump default가 set/enum/torch dtype을 직렬화하지 못해 안전한 캐스팅 추가
    with open(run_details_path, 'w', encoding='utf-8') as f:
        json.dump(
            run_details,
            f,
            indent=2,
            default=lambda o: list(o) if isinstance(o, set) else str(o),
        )
    print(f"Run details (data + hyperparameters) saved to: {run_details_path}")


    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        # peft_config=lora_config,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)], # EarlyStoppingCallback 활성화
    )

    print(f"Starting training for {version_name} with Wandb...")
    trainer.train()
    print(f"Training finished for {version_name}!")

    # 모델 저장 (LoRA 어댑터만 저장)
    trainer.save_model(current_output_dir)
    print(f"Model saved to: {current_output_dir}")

    # Wandb 런 종료
    wandb.finish()
    
    # 학습이 끝난 모델 메모리에서 해제
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

# --- 학습 실행 ---
# 학습률 우선 루프, 각 lr마다 모든 데이터셋을 순회
require_config("model_id", model_id)
require_config("wandb_project", wandb_project)
require_config("dataset_paths", dataset_paths)
require_config("base_output_dir", base_output_dir)

for lr in learning_rates:
    for dataset_path in dataset_paths:
        require_config("dataset_path", dataset_path)
        dataset_input_name = os.path.basename(os.path.normpath(dataset_path)) or "dataset"
        base_version_name = dataset_input_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{base_version_name}_{timestamp}"

        train_model_for_version(version_name, dataset_path, lr)

print("\n--- All automated training runs completed! ---")
