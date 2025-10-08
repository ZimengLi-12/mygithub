#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Pure Pursuit Controller Integration Script

This script loads the Qwen2.5-0.5B-Instruct model, loads a Few-Shot
context from the generated pp_llm.ndjson dataset, and implements robust 
decoding by forcing the model into an I/O pattern.

Key Fixes:
1. CUDA OOM solved by 4-bit quantization (BitsAndBytesConfig).
2. All instructions and context are placed in the 'user' role to suppress the 'assistant' tendency.
3. Explicit GenerationConfig is used to ensure deterministic (non-random) output.
4. Robust output parsing handles both 'Output: <value>' and '<value>' formats.
5. attention_mask is manually passed to suppress warnings and ensure correctness.
"""

import os
import json
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from typing import List, Dict, Union

# --- Configuration ---
N_WAYPOINTS = 20
NDJSON_PATH = "pp_llm.ndjson"
MAX_EXAMPLES = 15  # 保持为15，避免超长Prompt
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# --- 1. Prompt Definition and Context Loading (强化 I/O 模式) ---

SYSTEM_INSTRUCTIONS_HEADER = f"""
你是自动驾驶车辆的控制器，以固定频率运行。
你的任务是根据车辆当前状态和前方路径点信息，推理出前轮的期望转向角（度）。

车辆的输入信息（send_data）是一个JSON对象，包含以下字段：
1. "psi_state"：车辆当前航向角（度）。
2. "x_state", "y_state"：车辆当前位置（米）。
3. "waypoints"：一个包含未来 {N_WAYPOINTS} 个路径点的列表，每个点为 {{"x": <米>, "y": <米>}}。
4. "n_waypoints"：路径点数量，即 {N_WAYPOINTS}。

你的输出**必须**是车辆前轮的转向角（度），且**只**输出一行，以"Output:"开头，后跟一个浮点数字。
输出的格式必须严格为：Output: <转向角_度>
转向角的正值表示左转，负值表示右转。

请参考以下历史输入和输出模式，预测接下来的转向角。
输出的趋势应该保持平滑，以确保平稳驾驶。

--- 历史模式开始 (Few-Shot Examples) ---
"""
SYSTEM_INSTRUCTIONS_HEADER = textwrap.dedent(SYSTEM_INSTRUCTIONS_HEADER).strip()


def load_prompt_history(ndjson_path: str, max_examples: int) -> str:
    """将系统指令和 Few-Shot 示例组合成一个单一的 User Prompt 内容。"""
    if not os.path.exists(ndjson_path):
        print(f"Error: NDJSON file not found at {ndjson_path}. Using empty history.")
        return SYSTEM_INSTRUCTIONS_HEADER + "\n"
        
    history_lines = [SYSTEM_INSTRUCTIONS_HEADER]
    
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            try:
                data = json.loads(line)
                # LLM 输入
                input_msg = f"Input: {json.dumps(data['send_data'], ensure_ascii=False)}"
                # PP 算法的期望输出作为 LLM 的标签
                output_msg = f"Output: {data['expected_output_deg']}"
                history_lines.append(f"{input_msg}\n{output_msg}")
            except json.JSONDecodeError:
                print(f"Warning: Skipped malformed JSON line in {ndjson_path}")
                continue
            
    return "\n".join(history_lines) + "\n"

# 全局上下文历史
CONTEXT_HISTORY_TEXT = load_prompt_history(NDJSON_PATH, MAX_EXAMPLES)


# --- 2. 模型加载 (解决 CUDA OOM) ---

def load_qwen_model():
    """加载 Qwen2.5-0.5B-Instruct 模型和分词器，使用 4-bit 量化。"""
    print(f"Loading model: {MODEL_NAME} with 4-bit quantization...")
    
    # 4-bit 量化配置，解决 CUDA OOM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.eval()
        print("Model loaded successfully using 4-bit quantization.")
        return model, tokenizer
    except Exception as e:
        print(f"FATAL ERROR: Could not load model {MODEL_NAME} in quantized mode.")
        print(f"Details: {e}")
        return None, None

model, tokenizer = load_qwen_model()


# --- 3. 推理函数 (解决解析和助手模式) ---

def llm_query_wuen(new_send_data: Dict[str, Union[float, List[Dict]]]) -> float:
    """
    接收实时车辆状态数据，调用 Qwen 模型进行推理，并解析输出的转向角。
    """
    if model is None or tokenizer is None:
        return 0.0

    # 1. 格式化当前时间步的输入
    current_input_line = f"\n### 任务：预测下一个转向角 (度)\nInput: {json.dumps(new_send_data, ensure_ascii=False)}\nOutput:"
    
    # 2. 构造 messages: 移除 System 角色，将所有内容作为单一 User 消息
    # 强制模型进入 I/O 模式，避免助手回复
    messages = [
        {"role": "user", "content": CONTEXT_HISTORY_TEXT + current_input_line} 
    ]
    
    # 3. 转换为模型输入 ID (已修复注意力掩码警告)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 4. 手动创建 attention_mask
    attention_mask = input_ids.ne(tokenizer.eos_token_id).to(model.device)
    
    # 5. 生成配置 (确保零随机性，避免警告)
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(
        max_new_tokens=50,
        do_sample=False,
        num_beams=1,
        temperature=0.01, # 极低温度，避免随机生成
        top_p=1.0, 
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 6. 模型推理
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    # 7. 解码并提取 LLM 输出
    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # 8. 强化解析逻辑 (解决输出格式错误)
    # 尝试分离前缀，如果不存在前缀，则整个响应即为值
    if response.lower().startswith("output:"):
        # 模式 1: 有前缀
        value_str = response.split(":", 1)[-1].split('\n')[0].strip()
    else:
        # 模式 2: 纯值模式（取第一行）
        value_str = response.split('\n')[0].strip()

    try:
        steering_deg = float(value_str)
        return steering_deg
    except ValueError:
        print(f"Warning: Failed to parse float from LLM output. Raw output was NOT a number: '{response}'")
        return 0.0


# --- Main Execution / Simulation Example ---

if __name__ == "__main__":
    if model and tokenizer:
        print("\n--- Running Simulation Example (Reading 56th record from NDJSON) ---")
        
        # 从 NDJSON 文件中读取一个实际的输入和期望输出进行模拟测试
        test_data = None
        test_expected_output = None
        test_time = None
        
        try:
            with open(NDJSON_PATH, 'r', encoding='utf-8') as f:
                # 使用第 56 条记录（索引 55）进行测试
                lines = f.readlines()
                if len(lines) > 55:
                    test_record = json.loads(lines[55])
                    test_data = test_record['send_data']
                    test_expected_output = test_record['expected_output_deg']
                    test_time = test_record['t']
        except Exception as e:
            print(f"Could not read test data from {NDJSON_PATH}: {e}")
            exit()
            
        if test_data:
            print(f"Testing LLM with input from t={test_time}...")
            
            # 运行推理
            predicted_steering = llm_query_wuen(test_data)

            print("\n" + "="*50)
            print("SIMULATION RESULTS:")
            print(f"Input Time (t):                      {test_time}")
            print(f"Input State (x, y):                  ({test_data['x_state']}, {test_data['y_state']})")
            print(f"PP Ground Truth (Expected Steering): {test_expected_output:.7f} degrees")
            print(f"LLM Predicted Steering:              {predicted_steering:.7f} degrees")
            print("="*50)
        else:
            print(f"\nNot enough data in NDJSON (need > 55 records) to run simulation example. Please check file content.")