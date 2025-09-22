# ================= NEW LOCAL REWARD: Boundary Helpfulness =================
# y_phi == 1 for synthetic tuples; "wrong" means alpha predicts majority.

# ---- Banding by Alpha's distance to boundary (0.5) ----
margin = torch.abs(p1_phi_alpha - 0.5)   # [T]
tau_boundary = 0.05
tau_shoulder = 0.15

near_mask     = (margin <= tau_boundary)
shoulder_mask = (margin >  tau_boundary) & (margin <= tau_shoulder)
far_mask      = (margin >  tau_shoulder)

# ---- Realism gate with band-aware floor ----
floor_near, floor_shoulder, floor_far = 0.35, 0.20, 0.05
floor_per = torch.where(
    near_mask, torch.tensor(floor_near, device=margin.device),
    torch.where(shoulder_mask, torch.tensor(floor_shoulder, device=margin.device),
                torch.tensor(floor_far, device=margin.device))
)
realism_gate = torch.maximum(judge_conf, floor_per)  # baseline, in [0,1]

# ---- Boundary usefulness weights (driver) ----
alpha_wrong = (p1_phi_alpha < 0.5).float()  # y=1 synthetic

w_near_wrong, w_near_right         = 1.00, 0.40
w_shoulder_wrong, w_shoulder_right = 0.70, 0.20
w_far_any                           = 0.05

weight_near = near_mask.float() * (alpha_wrong * w_near_wrong + (1.0 - alpha_wrong) * w_near_right)
weight_shou = shoulder_mask.float() * (alpha_wrong * w_shoulder_wrong + (1.0 - alpha_wrong) * w_shoulder_right)
weight_far  = far_mask.float() * w_far_any
boundary_usefulness = (weight_near + weight_shou + weight_far).clamp(0.0, 1.0)

# ---- New local score: additive bonus on top of judge_conf ----
k_bonus = 0.6  # try 0.4–0.8 in small sweeps
score_local_raw = (realism_gate + k_bonus * (1.0 - realism_gate) * boundary_usefulness).clamp(0.0, 1.0)

LOCAL_CAP = 0.80
cap_t = torch.tensor(LOCAL_CAP, device=score_local_raw.device, dtype=score_local_raw.dtype)
over_mask = (score_local_raw > cap_t).float()
score_local = torch.minimum(score_local_raw, cap_t)

# Diagnostics (keep names your tracker expects)
uncert_alpha = 1.0 - (2.0 * margin).clamp(0, 1)   # boundary proximity
corr_factor  = boundary_usefulness                 # report the boundary weight