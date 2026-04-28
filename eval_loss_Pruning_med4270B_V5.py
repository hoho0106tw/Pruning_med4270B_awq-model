
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM

# =========================
# config
# =========================
FP16_MODEL = "m42-health/Llama3-Med42-70B"
AWQ_MODEL  = "hoho0106tw/Femh_Pruning_med4270B_awq-model"   
EXCEL_PATH = "sample_200_v4.xlsx"

DEVICE = "cuda"
MAX_LEN = 512
MIN_LEN = 100

# =========================
# tokenizer（固定用原始）
# =========================
tokenizer = AutoTokenizer.from_pretrained(FP16_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# build test data（不 chunk）
# =========================
def build_test_data():
    df = pd.read_excel(EXCEL_PATH)

    texts = []

    for _, row in df.iterrows():
        parts = []
        for col in ["S", "O", "A", "P"]:
            val = str(row[col]).strip()
            if val and val != "nan":
                parts.append(f"{col}: {val}")

        if not parts:
            continue

        text = " ".join(parts)

        ids = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=False
        )["input_ids"]

        if len(ids) < MIN_LEN:
            continue

        text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text)

    print(f"Test samples: {len(texts)}")
    return texts[:10]   

# =========================
# compute loss
# =========================
def compute_loss(model, texts):
    losses = []

    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, ppl

# =========================
# main
# =========================
test_texts = build_test_data()

# =========================
# FP16 評估（先跑）
# =========================
print("\n=== Loading FP16 model ===")
fp16_model = AutoModelForCausalLM.from_pretrained(
    FP16_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

print("\n=== Evaluating FP16 ===")
fp16_loss, fp16_ppl = compute_loss(fp16_model, test_texts)

print(f"FP16 Loss: {fp16_loss:.4f}, PPL: {fp16_ppl:.2f}")

# 🔥 釋放記憶體（關鍵）
del fp16_model
torch.cuda.empty_cache()

# =========================
# AWQ 評估（再跑）
# =========================
print("\n=== Loading AWQ model ===")
awq_model = AutoAWQForCausalLM.from_quantized(
    AWQ_MODEL,
    device_map="auto",
    trust_remote_code=True   # ✅ 建議加
).eval()

print("\n=== Evaluating AWQ ===")
awq_loss, awq_ppl = compute_loss(awq_model, test_texts)

print(f"AWQ Loss: {awq_loss:.4f}, PPL: {awq_ppl:.2f}")

# =========================
# compare
# =========================
print("\n===== RESULT =====")

delta = awq_loss - fp16_loss

print(f"FP16 Loss: {fp16_loss:.4f}")
print(f"AWQ  Loss: {awq_loss:.4f}")
print(f"Loss Δ: {delta:.4f}")

if delta < 0.05:
    print("🔥 Near-lossless quantization")
elif delta < 0.15:
    print("✅ Excellent quantization")
elif delta < 0.3:
    print("👍 Good quantization")
else:
    print("⚠️ Degradation detected")

