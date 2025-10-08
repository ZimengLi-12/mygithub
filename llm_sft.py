import os
import json
import textwrap
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from typing import List, Dict, Union 


# --- Configuration ---
NDJSON_PATH = "pp_llm.ndjson"
OUTPUT_DIR = "./qwen_pp_sft"
N_WAYPOINTS = 20
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# --- 1. 数据准备函数 ---

def format_data_for_sft(ndjson_path: str, max_examples: int) -> List[Dict[str, str]]:
    """读取 NDJSON，并将其转换为 SFTTrainer 期望的对话/指令格式。"""
    
    TASK_INSTRUCTION = f"""
    你是自动驾驶车辆的控制器。你的任务是根据车辆当前状态和前方路径点信息，
    推理出前轮的期望转向角（度）。

    输入 (Input) 是一个 JSON 对象，包含车辆状态和未来 {N_WAYPOINTS} 个路径点。
    你的输出 (Output) 必须是单个浮点数字符串，且以 'Output: ' 开头。
    请严格遵循以下格式。

    """
    TASK_INSTRUCTION = textwrap.dedent(TASK_INSTRUCTION).strip() + "\n\n"

    sft_data = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    try:
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if i >= max_examples:
                break
                
            data = json.loads(line)
            
            send_data_str = json.dumps(data['send_data'], ensure_ascii=False)
            assistant_message = f"Output: {data['expected_output_deg']}"
            
            formatted_dialogue = [
                {"role": "user", "content": f"{TASK_INSTRUCTION}Input: {send_data_str}"},
                {"role": "assistant", "content": assistant_message}
            ]
            
            full_text = tokenizer.apply_chat_template(
                formatted_dialogue,
                tokenize=False,
                add_generation_prompt=False
            )
            
            sft_data.append({"text": full_text})
            
    except Exception as e:
        print(f"Error processing data for SFT: {e}")
        return []

    return sft_data


# --- 2. QLoRA 和模型配置 (核心内存优化区) ---

def setup_qlora_training(train_dataset: Dataset, eval_dataset: Dataset):
    """配置模型、LoRA 和 Trainer，应用内存优化。"""
    
    # 4-bit 量化配置 (QLoRA)，解决 CUDA OOM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # ⬇️ 内存优化 1：强制关闭 use_cache，以兼容梯度检查点 ⬇️
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 训练参数 (已修复评估策略错误)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,         
        
        # ⬇️ 内存优化 2：减小批次大小，增加梯度累积步长 ⬇️
        per_device_train_batch_size=2,  # 从 4 减小到 2
        gradient_accumulation_steps=8,  # 从 4 增加到 8 (有效批次大小仍为 16)
        
        optim="paged_adamw_8bit",   
        logging_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_strategy="epoch",        
        evaluation_strategy="epoch",  
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        
        # ⬇️ 内存优化 3：显式启用梯度检查点 ⬇️
        gradient_checkpointing=True, 
        
        ddp_find_unused_parameters=False,
    )
    
    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,       
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=512, 
    )
    
    return trainer

# --- 3. 主程序执行流程 ---

def main_sft():
    # 假设您的 pp_llm.ndjson 足够大，我们使用前 300 条记录来训练
    MAX_RECORDS_FOR_TRAIN = 300 
    
    print("--- 1. Data Preparation ---")
    sft_records = format_data_for_sft(NDJSON_PATH, MAX_RECORDS_FOR_TRAIN)
    if not sft_records:
        print("Error: Training data preparation failed.")
        return

    train_dataset_full = Dataset.from_list(sft_records)
    
    if len(train_dataset_full) < 10:
        print("Error: Not enough records for train/test split. Need at least 10.")
        return

    # 拆分数据集：90% 用于训练，10% 用于评估 
    split_datasets = train_dataset_full.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]  
    
    print(f"Total Records Loaded: {len(train_dataset_full)}")
    print(f"Training Samples: {len(train_dataset)}, Evaluation Samples: {len(eval_dataset)}")
    
    
    print("--- 2. Model and Trainer Setup (QLoRA) ---")
    trainer = setup_qlora_training(train_dataset, eval_dataset)

    print("--- 3. Training Start ---")
    trainer.train()

    # 保存最终的 Adapter 权重
    final_output_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_output_path)
    print(f"\n✅ Training complete. Final adapter saved to: {final_output_path}")

    print("\n--- Next Step ---")
    print("您需要修改推理脚本 (llm_pure_pursuit_controller.py)，从这个路径加载微调后的 LoRA 权重，")
    print(f"才能获得高精度的转向角预测结果: {final_output_path}")


if __name__ == "__main__":
    if not os.path.exists(NDJSON_PATH):
        print(f"Error: {NDJSON_PATH} not found. Please ensure it exists before running SFT.")
    else:
        main_sft()