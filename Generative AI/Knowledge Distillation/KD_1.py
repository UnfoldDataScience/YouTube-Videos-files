import torch
import torch.nn.functional as F
from torch import optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config 
TEACHER = "gpt2-medium"
STUDENT = "gpt2"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
LR      = 5e-5
EPOCHS  = 2
TEMP    = 2.0      # temperature for soft targets

# Tiny dataset; 
prompts = [
    "The capital of France is",
    "Translate to French: Hello",
    "Once upon a time in a distant land",
    "In machine learning, overfitting occurs when",
    "Python is great for data science because",
]

# --- Load models/tokenizer ---
tok = AutoTokenizer.from_pretrained(STUDENT)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

teacher = AutoModelForCausalLM.from_pretrained(TEACHER).to(DEVICE).eval()
for p in teacher.parameters():  # freeze teacher
    p.requires_grad_(False)

student = AutoModelForCausalLM.from_pretrained(STUDENT).to(DEVICE).train()
opt = optim.AdamW(student.parameters(), lr=LR)

def kd_loss(student_logits, teacher_logits, attn_mask, T=2.0):
    """Response-based KD with correct KL signature and token shift + mask."""

    #drops the last time step’s logits because there’s no “next token” 
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]

    #attn_mask is 1 for real tokens, 0 for padding. We shift it to align with the target positions 
    m = attn_mask[:, 1:].float()

    B, S, V = s.shape #B- Batch Size, S = sequence length after shift, V= vocab size
    s = s.reshape(B * S, V)
    t = t.reshape(B * S, V)
    m = m.reshape(B * S)

    #Apply temperature scaling - note log_softmax and softmax
    #Higher T (>1) → softer, more uniform distribution.
    #Lower T (<1) → sharper, more confident distribution.
    log_p_s = F.log_softmax(s / T, dim=-1)  # log probs (student)
    p_t     = F.softmax(t / T, dim=-1)      # probs (teacher)

    #Token-wise KL divergence between student and teacher distributions over the vocabulary.
    kl_per_tok = F.kl_div(log_p_s, p_t, reduction="none").sum(-1)  # [B*S]

    #Mask out padding using m so only real tokens contribute.
    kd = (kl_per_tok * m).sum() / m.sum().clamp_min(1.0)

    #make gradient magnitudes temperature-independent.
    return kd * (T ** 2)  # Hinton scaling

def encode_batch(texts, max_len=128):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in enc.items()}

for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for i in range(0, len(prompts), 2):  # mini-batches of 2
        batch = prompts[i:i+2]
        enc = encode_batch(batch)

        with torch.no_grad():
            t_logits = teacher(**{k: enc[k] for k in ("input_ids", "attention_mask")}).logits

        s_out = student(**{k: enc[k] for k in ("input_ids", "attention_mask")})
        s_logits = s_out.logits

        loss = kd_loss(s_logits, t_logits, enc["attention_mask"], T=TEMP)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch}: avg KD loss = {total / max(len(prompts)//2, 1):.4f}")

# Save distilled student
student.save_pretrained("distilled-student-kd-only")
tok.save_pretrained("distilled-student-kd-only")
print("Saved to distilled-student-kd-only/")
