import math
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_NAME = "gpt2-medium"
STUDENT_NAME = "distilled-student-kd-only"

EVAL_TEXTS = [
    "Paris is the capital of France.",
    "In machine learning, regularization helps to prevent",
    "Transformers are effective because they use",
    "Python is popular for data science due to",
]

def load_model_and_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
    return model, tok

@torch.no_grad()
def perplexity(texts, model, tok, max_len=128):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    enc["labels"] = enc["input_ids"].clone()
    out = model(**enc)
    loss = F.cross_entropy(
        out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
        enc["labels"][:, 1:].reshape(-1),
        ignore_index=tok.pad_token_id
    )
    return math.exp(loss.item())

def model_info(name):
    model = AutoModelForCausalLM.from_pretrained(name)
    params = sum(p.numel() for p in model.parameters())

    # If it's local, measure directly
    if os.path.isdir(name):
        local_path = name
    else:
        # Look inside HF cache folder for this model
        local_path = os.path.join(HF_HUB_CACHE, name.replace("/", "--"))
        if not os.path.exists(local_path):
            size_gb = 0.0
            return params, size_gb

    size_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(local_path)
        for f in files
    )
    size_gb = size_bytes / (1024**3)
    return params, size_gb

# Load models
teacher, tok_t = load_model_and_tokenizer(TEACHER_NAME)
student, tok_s = load_model_and_tokenizer(STUDENT_NAME)

# Stats & PPL
teacher_params, teacher_size = model_info(TEACHER_NAME)
student_params, student_size = model_info(STUDENT_NAME)

ppl_teacher = perplexity(EVAL_TEXTS, teacher, tok_t)
ppl_student = perplexity(EVAL_TEXTS, student, tok_s)

# Print table
print(f"{'Model':<20} {'Params (M)':<15} {'Disk Size (GB)':<15} {'PPL':<10}")
print("-"*60)
print(f"{'Teacher':<20} {teacher_params/1e6:<15.1f} {teacher_size:<15.2f} {ppl_teacher:<10.2f}")
print(f"{'Student':<20} {student_params/1e6:<15.1f} {student_size:<15.2f} {ppl_student:<10.2f}")

# Summary
size_reduction = 100 * (1 - student_params/teacher_params)
print(f"\nSize reduction: {size_reduction:.1f}% fewer parameters")