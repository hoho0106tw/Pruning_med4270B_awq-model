#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import gc
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM

# =========================
# CONFIG
# =========================
FP16_MODEL = "m42-health/Llama3-Med42-70B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med4270B_awq-model"   

EXCEL_PATH = "sample_200_v4.xlsx"

DEVICE = "cuda"
N_SAMPLES = 150
MAX_TOKENS = 512

LABELS = [
    "stroke","transient ischemic attack","dementia","epilepsy","migraine",
    "parkinsonism","neuropathy","radiculopathy","spine disease",
    "carotid artery disease","syncope"
]

WINDOWS = [
    ("S+O", ["S","O"]),
    ("O+A", ["O","A"]),
    ("A+P", ["A","P"]),
]

WINDOW_WEIGHTS = {
    "S+O": 0.6,
    "O+A": 1.2,
    "A+P": 2.5   
}

# =========================
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# =========================
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.strip()

# =========================
def match_label(pred):
    pred = normalize(pred)

    for label in sorted(LABELS, key=len, reverse=True):
        if label in pred:
            return label

    return None

# =========================
def build_windows(row):
    results = []

    for name, cols in WINDOWS:
        parts = []
        for col in cols:
            val = str(row[col]).strip()
            if val and val != "nan":
                parts.append(f"{col}: {val}")

        if parts:
            results.append((name, " ".join(parts)))

    return results

# =========================
def build_prompt(window_name, text):
    label_str = ", ".join(LABELS)

    return f"""
You are a clinical diagnosis classifier.

Choose EXACTLY ONE label from:
{label_str}

Rules:
- Output ONLY the label
- No explanation

Context: {window_name}

SOAP:
{text}

Answer:
""".strip()

# =========================
def tokenize_prompt(prompt, tokenizer):
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    )

# =========================
def infer_one(model, tokenizer, prompt):
    inputs = tokenize_prompt(prompt, tokenizer)
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            temperature=0.0
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    return text.strip()

# =========================
def infer_row(model, tokenizer, row):
    windows = build_windows(row)

    votes = Counter()
    details = []
    ap_pred = None

    for name, text in windows:
        prompt = build_prompt(name, text)
        pred_raw = infer_one(model, tokenizer, prompt)
        pred = match_label(pred_raw)

        details.append((name, pred, pred_raw[:80]))

        if name == "A+P" and pred is not None:
            ap_pred = pred

        if pred is not None:
            votes[pred] += WINDOW_WEIGHTS[name]

    if ap_pred is not None:
        return ap_pred, details

    if votes:
        return votes.most_common(1)[0][0], details

    return "stroke", details

# =========================
def evaluate(model, tokenizer, df, name):
    correct = 0

    for i, row in df.iterrows():
        gt = normalize(str(row["PRIMARY_DIAGNOSIS"]))
        pred, details = infer_row(model, tokenizer, row)

        if pred == gt:
            correct += 1

        print(f"[{name}] {i+1}/{len(df)}")
        print("GT:", gt)
        print("Pred:", pred)

        for d in details:
            print(f"  {d[0]} → {d[1]} | {d[2]}")

        print("----")

    acc = correct / len(df)
    print(f"\n{name} Accuracy: {acc:.3f}")

    return acc

# =========================
def main():

    df = pd.read_excel(EXCEL_PATH)
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # FP16
    print("Loading FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        FP16_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    print("\n=== FP16 ===")
    acc_fp16 = evaluate(model, tokenizer, df, "FP16")

    del model
    clean_memory()

    # AWQ
    print("Loading AWQ...")
    model = AutoAWQForCausalLM.from_quantized(
        AWQ_MODEL,
        device_map="auto",
        trust_remote_code=True  
    ).eval()

    print("\n=== AWQ ===")
    acc_awq = evaluate(model, tokenizer, df, "AWQ")

    print("\n===== FINAL =====")
    print("FP16:", acc_fp16)
    print("AWQ :", acc_awq)
    print("Δ   :", acc_awq - acc_fp16)

# =========================
if __name__ == "__main__":
    main()