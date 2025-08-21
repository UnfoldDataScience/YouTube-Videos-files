import torch
import torch.nn.functional as F

# --- distributions (teacher vs student) over 3 classes ---
p_t = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)          # teacher PROBABILITIES
p_s = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)          # student PROBABILITIES

# Convert student to log-probs (as F.kl_div expects for 'input')
log_p_s = torch.log(p_s)
# Also prepare teacher log-probs for the second variant
log_p_t = torch.log(p_t)

# ---- Variant A: teacher as PROBS (default: log_target=False) ----
kl_probs = F.kl_div(input=log_p_s, target=p_t, reduction="sum", log_target=False)

# ---- Variant B: teacher as LOG-PROBS (set log_target=True) ----
kl_logprobs = F.kl_div(input=log_p_s, target=log_p_t, reduction="sum", log_target=True)

print("KL with teacher as probs      :", kl_probs.item())
print("KL with teacher as log-probs  :", kl_logprobs.item())
print("Are they (almost) equal?      :", torch.allclose(kl_probs, kl_logprobs, atol=1e-7))
