import torch


# ------------------------------------------------------------------ loss

def bce_per_sample_from_probs(y_true, p1, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample binary cross-entropy from P(y=1). Returns [N] tensor."""
    device = p1.device
    y   = y_true.to(device).float()
    eps = max(1e-8, float(torch.finfo(p1.dtype).eps))
    p   = torch.clamp(p1, eps, 1.0 - eps)
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def group_mean_loss(losses: torch.Tensor, a, group_value: int) -> torch.Tensor:
    """Mean loss for one group. Returns NaN if group has no samples."""
    a    = torch.as_tensor(a, device=losses.device).long()
    mask = (a == int(group_value))
    if mask.sum().item() == 0:
        return torch.tensor(float("nan"), device=losses.device)
    return losses[mask].mean()


def worst_group_loss(losses: torch.Tensor, a, group_values=(0, 1)):
    """
    Per-group mean losses and the worst (max) across present groups.
    Returns: (worst_loss tensor scalar, {g: mean_loss float-or-nan})
    """
    per_g, vals = {}, []
    for g in group_values:
        mg = group_mean_loss(losses, a, g)
        per_g[int(g)] = float(mg.item()) if (mg == mg) else float("nan")
        if mg == mg:
            vals.append(mg)
    if not vals:
        return torch.tensor(float("nan"), device=losses.device), per_g
    return torch.stack(vals).max(), per_g


# ------------------------------------------------------------------ inference

def p1_from_agent(agent, x: torch.Tensor, *, no_grad: bool = True) -> torch.Tensor:
    """P(y=1|x) from an FFNNAgent. Shape [...,]."""
    agent.model.eval()
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        return torch.softmax(agent.model(x), dim=-1)[..., 1]


# ------------------------------------------------------------------ metrics

def f1_from_probs(y_true, p1: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary F1 for the positive class."""
    y_true = y_true.to(p1.device).long()
    y_pred = (p1 >= threshold).long()
    eps    = 1e-8
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)


def acc_from_probs(y_true, p1: torch.Tensor, threshold: float = 0.5) -> float:
    """Binary accuracy."""
    y_true = y_true.to(p1.device).long()
    return float(((p1 >= threshold).long() == y_true).float().mean().item())


def roc_auc_from_probs(y_true, p1: torch.Tensor) -> float:
    """ROC-AUC. Returns NaN if only one class present."""
    try:
        from sklearn.metrics import roc_auc_score
        y_np = y_true.cpu().numpy().astype(int)
        p_np = p1.detach().cpu().numpy()
        return float("nan") if len(set(y_np)) < 2 else float(roc_auc_score(y_np, p_np))
    except Exception:
        return float("nan")


def _safe_rate(num, den) -> float:
    d = float(den.item()) if torch.is_tensor(den) else float(den)
    n = float(num.item()) if torch.is_tensor(num) else float(num)
    return float("nan") if d <= 0 else n / d


# ------------------------------------------------------------------ fairness

def soft_eo_gap(a, y_true, p1: torch.Tensor) -> torch.Tensor:
    """
    Soft (threshold-free) EO gap: |E[p1 | y=1, a=0] - E[p1 | y=1, a=1]|.
    Returns NaN if a group has no positive samples.
    """
    device  = p1.device
    a01     = (torch.as_tensor(a, device=device).long() != 0).long()
    pos     = y_true.to(device).long() == 1
    means   = {}
    for g in (0, 1):
        gp = (a01 == g) & pos
        if gp.sum().item() == 0:
            return torch.tensor(float("nan"), device=device)
        means[g] = p1[gp].mean()
    return torch.abs(means[0] - means[1])


def fairness_classification_metrics(
    a, y_true, p1: torch.Tensor, *, threshold: float = 0.5
) -> dict:
    """
    DP, EO (TPR diff), EOd for binary protected attribute.
    Returns NaN for any metric if a group is absent.
    Includes per-group rates for detailed logging.
    """
    device = p1.device
    y_true = y_true.to(device).long()
    a01    = (torch.as_tensor(a, device=device).long() != 0).long()
    y_hat  = (p1 >= threshold).long()

    dp, tpr, fpr = {}, {}, {}
    for g in (0, 1):
        m   = (a01 == g)
        pos = m & (y_true == 1)
        neg = m & (y_true == 0)
        dp[g]  = _safe_rate(((y_hat == 1) & m).sum(),   m.sum())
        tpr[g] = _safe_rate(((y_hat == 1) & pos).sum(), pos.sum())
        fpr[g] = _safe_rate(((y_hat == 1) & neg).sum(), neg.sum())

    def _absdiff(x, y):
        return float("nan") if (x != x or y != y) else abs(x - y)

    tpr_diff = _absdiff(tpr[0], tpr[1])
    fpr_diff = _absdiff(fpr[0], fpr[1])
    both_ok  = (tpr_diff == tpr_diff) and (fpr_diff == fpr_diff)

    return {
        "dp_diff":      _absdiff(dp[0], dp[1]),
        "eo_tpr_diff":  tpr_diff,
        "eod_fpr_diff": fpr_diff,
        "eod_max_diff": max(tpr_diff, fpr_diff)        if both_ok else float("nan"),
        "eod_avg_diff": 0.5 * (tpr_diff + fpr_diff)   if both_ok else float("nan"),
        "dp_rate_g0":   dp[0],  "dp_rate_g1":  dp[1],
        "tpr_g0":       tpr[0], "tpr_g1":      tpr[1],
        "fpr_g0":       fpr[0], "fpr_g1":      fpr[1],
    }


def disadvantaged_group_from_alpha(
    alpha_model, x_val: torch.Tensor, y_val: torch.Tensor, a_val,
    *, group_values=(0, 1),
) -> tuple[int, int, dict, float]:
    """
    Returns (disadv, adv, per_g, worst) where disadv is the group with higher
    mean BCE under alpha.
    """
    with torch.no_grad():
        p1     = p1_from_agent(alpha_model, x_val)
        losses = bce_per_sample_from_probs(y_val.long(), p1)
        worst_t, per_g = worst_group_loss(losses, a_val, group_values=group_values)

    lg0, lg1 = per_g.get(0, float("nan")), per_g.get(1, float("nan"))
    if lg0 != lg0:
        disadv = 1
    elif lg1 != lg1:
        disadv = 0
    else:
        disadv = 0 if lg0 > lg1 else 1

    worst = float(worst_t.item()) if (worst_t == worst_t) else float("nan")
    return disadv, 1 - disadv, per_g, worst
