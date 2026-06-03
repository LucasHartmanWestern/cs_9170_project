import csv
import json
import numpy as np
import torch
from pathlib import Path


class TestSuite:
    def __init__(self, seed_dir, beta_factory=None, best_beta_path=None, dataset=None, **_ignored):
        self.seed_dir = Path(seed_dir)
        self.beta_factory = beta_factory
        self.best_beta_path = Path(best_beta_path) if best_beta_path else None
        self.dataset = dataset

    # ------------------------------------------------------------------ helpers

    def _safe_rate(self, num, den) -> float:
        d = float(den.item()) if torch.is_tensor(den) else float(den)
        return float("nan") if d <= 0 else float((num.float() / den.float()).item())

    def _p1_from_agent(self, agent, x) -> torch.Tensor:
        agent.model.eval()
        with torch.no_grad():
            return torch.softmax(agent.model(x), dim=-1)[..., 1]

    def _all_f1_from_probs(self, y_true, p1, threshold=0.5):
        eps = 1e-8
        y_bin = y_true.long().to(p1.device)
        y_hat = (p1 >= threshold).long()

        tp1 = ((y_hat == 1) & (y_bin == 1)).sum().float()
        fp1 = ((y_hat == 1) & (y_bin == 0)).sum().float()
        fn1 = ((y_hat == 0) & (y_bin == 1)).sum().float()
        prec1 = tp1 / (tp1 + fp1 + eps)
        rec1  = tp1 / (tp1 + fn1 + eps)
        f1_1  = 2 * prec1 * rec1 / (prec1 + rec1 + eps)

        tp0 = ((y_hat == 0) & (y_bin == 0)).sum().float()
        fn0 = ((y_hat == 1) & (y_bin == 0)).sum().float()
        fp0 = ((y_hat == 0) & (y_bin == 1)).sum().float()
        prec0 = tp0 / (tp0 + fn0 + eps)
        rec0  = tp0 / (tp0 + fp0 + eps)
        f1_0  = 2 * prec0 * rec0 / (prec0 + rec0 + eps)

        n1 = (y_bin == 1).sum().float()
        n0 = (y_bin == 0).sum().float()
        n  = n0 + n1 + eps
        f1_w = float(n1 * f1_1 / n + n0 * f1_0 / n)
        return float(f1_1), float(f1_0), f1_w, float(0.5 * (f1_1 + f1_0))

    def _brier_mean(self, y_true, p1) -> float:
        return float(((p1 - y_true.float().to(p1.device)) ** 2).mean())

    def _acc_from_probs(self, y_true, p1, threshold=0.5) -> float:
        y_bin = y_true.long().to(p1.device)
        return float(((p1 >= threshold).long() == y_bin).float().mean())

    def _roc_auc_from_probs(self, y_true, p1) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            y_np = y_true.long().cpu().numpy().astype(int)
            p_np = p1.detach().cpu().numpy()
            return float("nan") if len(set(y_np)) < 2 else float(roc_auc_score(y_np, p_np))
        except Exception:
            return float("nan")

    def _fairness_from_probs(self, y_true, p1, a, threshold=0.5) -> dict:
        device = p1.device
        y_bin = y_true.long().to(device)
        y_hat = (p1 >= threshold).long()
        a01   = (torch.as_tensor(a, device=device).long() != 0).long()

        dp_rates, tpr, fpr = {}, {}, {}
        for g in (0, 1):
            m   = a01 == g
            pos = m & (y_bin == 1)
            neg = m & (y_bin == 0)
            dp_rates[g] = self._safe_rate(((y_hat == 1) & m).sum(),   m.sum())
            tpr[g]      = self._safe_rate(((y_hat == 1) & pos).sum(), pos.sum())
            fpr[g]      = self._safe_rate(((y_hat == 1) & neg).sum(), neg.sum())

        def _absdiff(x, y):
            return float("nan") if (x != x or y != y) else abs(x - y)

        tpr_diff = _absdiff(tpr[0], tpr[1])
        fpr_diff = _absdiff(fpr[0], fpr[1])
        both_ok  = (tpr_diff == tpr_diff) and (fpr_diff == fpr_diff)
        return {
            "dp_diff":      _absdiff(dp_rates[0], dp_rates[1]),
            "eo_tpr_diff":  tpr_diff,
            "eod_fpr_diff": fpr_diff,
            "eod_max_diff": max(tpr_diff, fpr_diff) if both_ok else float("nan"),
            "eod_avg_diff": 0.5 * (tpr_diff + fpr_diff) if both_ok else float("nan"),
            "dp_rate_g0":   dp_rates[0], "dp_rate_g1": dp_rates[1],
            "tpr_g0":       tpr[0],      "tpr_g1":     tpr[1],
            "fpr_g0":       fpr[0],      "fpr_g1":     fpr[1],
        }

    # ---------------------------------------------------------------- main eval

    def log_final_test(
        self,
        alpha_model,
        x_test,
        y_test,
        f1_thresh: float = 0.5,
        prefer_best_beta: bool = True,
        beta_model=None,
        a_test=None,
    ):
        # Load best beta checkpoint if available
        beta = None
        phase1_path = self.seed_dir / "best_beta_state_dict_phase1_class1.pt"
        best_path   = phase1_path if phase1_path.exists() else self.best_beta_path
        if prefer_best_beta and self.beta_factory and best_path and best_path.exists():
            beta = self.beta_factory()
            beta.model.load_state_dict(torch.load(best_path, map_location=x_test.device))
            print(f"[TestSuite] Loaded best β from: {best_path}")
        elif beta_model is not None:
            beta = beta_model
            print("[TestSuite] Using provided β model.")
        else:
            print("[TestSuite] No β available for test evaluation.")

        # Alpha
        with torch.no_grad():
            p_alpha = self._p1_from_agent(alpha_model, x_test)
        a_f1_min, a_f1_maj, a_f1_w, a_f1_macro = self._all_f1_from_probs(y_test, p_alpha, f1_thresh)

        # Beta
        p_beta = None
        if beta is not None:
            with torch.no_grad():
                p_beta = self._p1_from_agent(beta, x_test)
            b_f1_min, b_f1_maj, b_f1_w, b_f1_macro = self._all_f1_from_probs(y_test, p_beta, f1_thresh)
        else:
            b_f1_min = b_f1_maj = b_f1_w = b_f1_macro = float("nan")

        _nan_fair = {k: float("nan") for k in (
            "dp_diff", "eo_tpr_diff", "eod_fpr_diff", "eod_max_diff", "eod_avg_diff",
            "dp_rate_g0", "dp_rate_g1", "tpr_g0", "tpr_g1", "fpr_g0", "fpr_g1",
        )}
        if a_test is None:
            print("[TestSuite] a_test not provided; fairness metrics will be NaN.")
            a_fair = b_fair = _nan_fair
        else:
            a_fair = self._fairness_from_probs(y_test, p_alpha, a_test, f1_thresh)
            b_fair = self._fairness_from_probs(y_test, p_beta, a_test, f1_thresh) if p_beta is not None else _nan_fair

        a_brier = self._brier_mean(y_test, p_alpha)
        a_acc   = self._acc_from_probs(y_test, p_alpha, f1_thresh)
        a_auc   = self._roc_auc_from_probs(y_test, p_alpha)
        b_brier = self._brier_mean(y_test, p_beta) if p_beta is not None else float("nan")
        b_acc   = self._acc_from_probs(y_test, p_beta, f1_thresh) if p_beta is not None else float("nan")
        b_auc   = self._roc_auc_from_probs(y_test, p_beta) if p_beta is not None else float("nan")

        print(f"[TEST] α -> F1w={a_f1_w:.4f}  AUC={a_auc:.4f}  EO={a_fair['eo_tpr_diff']:.4f}")
        print(f"[TEST] β -> F1w={b_f1_w:.4f}  AUC={b_auc:.4f}  EO={b_fair['eo_tpr_diff']:.4f}")

        row = {
            "alpha_f1_minority": a_f1_min, "alpha_f1_majority": a_f1_maj,
            "alpha_f1_weighted": a_f1_w,   "alpha_f1_macro":    a_f1_macro,
            "alpha_brier": a_brier, "alpha_acc": a_acc, "alpha_auc": a_auc,
            **{f"alpha_{k}": v for k, v in a_fair.items()},
            "beta_f1_minority": b_f1_min, "beta_f1_majority": b_f1_maj,
            "beta_f1_weighted": b_f1_w,   "beta_f1_macro":    b_f1_macro,
            "beta_brier": b_brier, "beta_acc": b_acc, "beta_auc": b_auc,
            **{f"beta_{k}": v for k, v in b_fair.items()},
        }

        out_path = self.seed_dir / "final_test_metrics.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"[TestSuite] Final test metrics → {out_path}")
        return row
