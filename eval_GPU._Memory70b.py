
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

# =========================
# config
# =========================
FP16_MODEL = "m42-health/Llama3-Med42-70B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med4270B_awq-model"   

# =========================
# tokenizer（用原始）
# =========================
tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# utils
# =========================
def to_gb(x):
    return x / 1024**3

def print_gpu_mem():
    print("\n[GPU usage]")
    for i in range(torch.cuda.device_count()):
        alloc = to_gb(torch.cuda.memory_allocated(i))
        reserved = to_gb(torch.cuda.memory_reserved(i))
        print(f"GPU {i}: allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")

# =========================
# 1️⃣ 權重記憶體（不 forward）
# =========================
def measure_weight_memory(load_fn, name):
    torch.cuda.empty_cache()

    print(f"\n===== [Weight] {name} =====")
    model = load_fn()

    print_gpu_mem()

    total = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
    total = to_gb(total)

    print(f"{name} Total Weight Memory: {total:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return total

# =========================
# 2️⃣ 推論記憶體（含 activation）
# =========================
def measure_inference_memory(load_fn, name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n===== [Inference] {name} =====")

    model = load_fn().eval()

    dummy = torch.randint(0, 1000, (1, 256)).to("cuda")

    with torch.no_grad():
        _ = model(input_ids=dummy)

    print_gpu_mem()

    total_peak = sum(
        torch.cuda.max_memory_allocated(i)
        for i in range(torch.cuda.device_count())
    )
    total_peak = to_gb(total_peak)

    print(f"{name} Total Inference Peak: {total_peak:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return total_peak

# =========================
# load functions
# =========================
def load_fp16():
    return AutoModelForCausalLM.from_pretrained(
        FP16_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

def load_awq():
    return AutoAWQForCausalLM.from_quantized(
        AWQ_MODEL,
        device_map="auto",
        trust_remote_code=True   
    )

# =========================
# run
# =========================

# ===== 權重 =====
fp16_w = measure_weight_memory(load_fp16, "FP16")
awq_w  = measure_weight_memory(load_awq,  "AWQ")

# ===== 推論 =====
fp16_i = measure_inference_memory(load_fp16, "FP16")
awq_i  = measure_inference_memory(load_awq,  "AWQ")

# =========================
# summary
# =========================
print("\n===== SUMMARY =====")

print("\n--- Weight Memory ---")
print(f"FP16: {fp16_w:.2f} GB")
print(f"AWQ : {awq_w:.2f} GB")
print(f"Reduction: {fp16_w / awq_w:.2f}x")

print("\n--- Inference Memory ---")
print(f"FP16: {fp16_i:.2f} GB")
print(f"AWQ : {awq_i:.2f} GB")
print(f"Reduction: {fp16_i / awq_i:.2f}x")