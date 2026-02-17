# reward_helpers.py
import torch

# ---------------- Loss helpers for Group DRO ----------------

def bce_per_sample_from_probs(y_true, p1, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample binary cross-entropy using probabilities p1 = P(y=1|x).
    Returns: [N] tensor
    """
    device = p1.device
    y = y_true.to(device).float()
    eps = max(1e-8, float(torch.finfo(p1.dtype).eps))
    p = torch.clamp(p1, eps, 1.0 - eps)
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def group_mean_loss(losses: torch.Tensor, a, group_value: int) -> torch.Tensor:
    """
    Mean of losses for group_value. Returns NaN if group has no samples.
    """
    device = losses.device
    a = torch.as_tensor(a, device=device).long()
    mask = (a == int(group_value))
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(float("nan"), device=device)
    return losses[mask].mean()


def worst_group_loss(losses: torch.Tensor, a, group_values=(0, 1)):
    """
    Computes per-group mean losses and the worst (max) across groups.
    If a group is missing (no samples), its mean is NaN and ignored in max.

    Returns:
      worst_loss (tensor scalar), per_g (dict {g: mean_loss (float or nan)})
    """
    device = losses.device
    per_g = {}
    vals = []

    for g in group_values:
        mg = group_mean_loss(losses, a, g)
        per_g[int(g)] = float(mg.item()) if (mg == mg) else float("nan")
        if mg == mg:  # not NaN
            vals.append(mg)

    if len(vals) == 0:
        return torch.tensor(float("nan"), device=device), per_g

    worst = torch.stack(vals).max()
    return worst, per_g


# ---------------- Reward helpers ----------------

def p1_from_agent(agent, x: torch.Tensor, *, no_grad: bool = True) -> torch.Tensor:
    """
    Generic "get P(y=1|x)" from an agent with .model that outputs logits for 2 classes.
    Returns probs for class 1 with shape [...,]
    """
    agent.model.eval()
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = agent.model(x)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1]


def f1_from_probs(y_true, p1: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binary F1 for the positive class, computed from probabilities and a threshold.
    Returns scalar tensor.
    """
    y_true = y_true.to(p1.device).long()
    y_pred = (p1 >= threshold).long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def brier_per_sample(y_true, p1: torch.Tensor) -> torch.Tensor:
    """
    Per-sample Brier score (MSE on probabilities).
    Returns: [N] tensor
    """
    y_true = y_true.to(p1.device).float()
    return (p1 - y_true) ** 2


# ---------------- Fairness (Equal Opportunity) helpers ----------------

def tpr_from_probs(a, y_true, p1: torch.Tensor, *, thresh: float = 0.5, group_value: int = 0) -> torch.Tensor:
    """
    True Positive Rate for a specific group:
    TPR(a=g) = P(ŷ=1 | y=1, A=g)
    Returns NaN if denominator is 0 (no positives for that group).
    """
    device = p1.device
    a = torch.as_tensor(a, device=device).long()
    y_true = y_true.to(device).long()
    y_pred = (p1 >= thresh).long()

    group_mask = (a == int(group_value))
    pos_mask = (y_true == 1)
    denom_mask = group_mask & pos_mask

    denom = denom_mask.sum().float()
    if denom.item() == 0:
        return torch.tensor(float("nan"), device=device)

    tp = ((y_pred == 1) & denom_mask).sum().float()
    return tp / (denom + 1e-8)


def eo_gap_from_probs(a, y_true, p1: torch.Tensor, *, thresh: float = 0.5, group0: int = 0, group1: int = 1) -> torch.Tensor:
    """
    Equal Opportunity gap:
    ΔEO = |TPR(A=group0) - TPR(A=group1)|
    Returns NaN if one of the groups has no positives (can't compute TPR).
    """
    tpr0 = tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group0)
    tpr1 = tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group1)

    if (tpr0 != tpr0) or (tpr1 != tpr1):
        return torch.tensor(float("nan"), device=p1.device)

    return torch.abs(tpr0 - tpr1)


def eo_signed_diff_from_probs(a, y_true, p1: torch.Tensor, *, thresh: float = 0.5, group0: int = 0, group1: int = 1) -> torch.Tensor:
    """
    Signed difference (useful for debugging directionality):
    TPR(group0) - TPR(group1)
    """
    tpr0 = tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group0)
    tpr1 = tpr_from_probs(a, y_true, p1, thresh=thresh, group_value=group1)
    if (tpr0 != tpr0) or (tpr1 != tpr1):
        return torch.tensor(float("nan"), device=p1.device)
    return (tpr0 - tpr1)


# reward_helpers.py

def disadvantaged_group_from_alpha(
    alpha_model,
    x_theta_val: torch.Tensor,
    y_theta_val: torch.Tensor,
    a_theta_val,
    *,
    group_values=(0, 1),
) -> tuple[int, int, dict, float]:
    """
    Returns:
      disadv (int): group id (0/1) with HIGHER mean BCE loss under alpha
      adv   (int): the other group
      per_g (dict): mean loss per group {0: float/nan, 1: float/nan}
      worst (float): worst-group mean loss (nan if can't compute)
    """
    with torch.no_grad():
        p1 = p1_from_agent(alpha_model, x_theta_val)                 # [N]
        losses = bce_per_sample_from_probs(y_theta_val.long(), p1)   # [N]
        worst_t, per_g = worst_group_loss(losses, a_theta_val, group_values=group_values)

    lg0 = per_g.get(0, float("nan"))
    lg1 = per_g.get(1, float("nan"))

    if lg0 != lg0:          # g0 missing
        disadv = 1
    elif lg1 != lg1:        # g1 missing
        disadv = 0
    else:
        disadv = 0 if lg0 > lg1 else 1

    adv = 1 - disadv
    worst = float(worst_t.item()) if (worst_t == worst_t) else float("nan")
    return disadv, adv, per_g, worst





def nearest_anchor_dist(x: torch.Tensor, anchors: torch.Tensor, *, chunk: int = 512) -> torch.Tensor:
    """
    Returns min Euclidean distance from each x[i] to any anchor.
    x:       [T, A]
    anchors: [N, A]
    output:  [T]
    VRAM-safe via chunking.
    """
    device = x.device
    anchors = anchors.to(device)

    T = x.shape[0]
    out = torch.empty((T,), device=device, dtype=x.dtype)

    a2 = (anchors * anchors).sum(dim=1)  # [N]

    for s in range(0, T, chunk):
        xb = x[s:s + chunk]  # [b, A]
        x2 = (xb * xb).sum(dim=1, keepdim=True)  # [b,1]
        d2 = x2 + a2.unsqueeze(0) - 2.0 * (xb @ anchors.t())
        d2 = torch.clamp(d2, min=0.0)
        out[s:s + chunk] = torch.sqrt(d2.min(dim=1).values + 1e-12)

    return out


def diversity_penalty(x: torch.Tensor, *, max_pts: int = 128, rho: float = 0.5) -> torch.Tensor:
    """
    Scalar penalty high when samples are too similar.
    Uses a subsample + gaussian similarity exp(-||xi-xj||^2/(2*rho^2)).
    """
    T = x.shape[0]
    if T <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    if T > max_pts:
        idx = torch.linspace(0, T - 1, steps=max_pts, device=x.device).long()
        Xd = x[idx]
    else:
        Xd = x

    D = torch.cdist(Xd, Xd)
    M = D.shape[0]
    D = D + torch.eye(M, device=D.device, dtype=D.dtype) * 1e6

    rho_t = torch.tensor(float(rho), device=D.device, dtype=D.dtype)
    sim = torch.exp(-0.5 * (D / (rho_t + 1e-8)) ** 2)
    return sim.mean()
