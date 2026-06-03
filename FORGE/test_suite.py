import csv
import sys
import numpy as np
import torch
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root), str(_root / "utilities"), str(_root / "FORGE")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reward_helpers as rh


class TestSuite:
    def __init__(self, seed_dir, beta_factory=None, best_beta_path=None, dataset=None, **_ignored):
        self.seed_dir = Path(seed_dir)
        self.beta_factory = beta_factory
        self.best_beta_path = Path(best_beta_path) if best_beta_path else None
        self.dataset = dataset

    # ------------------------------------------------------------------ metrics

    def _all_f1_from_probs(self, y_true, p1, threshold=0.5):
        """Returns (f1_minority, f1_majority, f1_weighted, f1_macro)."""
        eps   = 1e-8
        y_bin = y_true.long().to(p1.device)
        y_hat = (p1 >= threshold).long()

        tp1 = ((y_hat == 1) & (y_bin == 1)).sum().float()
        fp1 = ((y_hat == 1) & (y_bin == 0)).sum().float()
        fn1 = ((y_hat == 0) & (y_bin == 1)).sum().float()
        prec1 = tp1 / (tp1 + fp1 + eps); rec1 = tp1 / (tp1 + fn1 + eps)
        f1_1  = 2 * prec1 * rec1 / (prec1 + rec1 + eps)

        tp0 = ((y_hat == 0) & (y_bin == 0)).sum().float()
        fn0 = ((y_hat == 1) & (y_bin == 0)).sum().float()
        fp0 = ((y_hat == 0) & (y_bin == 1)).sum().float()
        prec0 = tp0 / (tp0 + fn0 + eps); rec0 = tp0 / (tp0 + fp0 + eps)
        f1_0  = 2 * prec0 * rec0 / (prec0 + rec0 + eps)

        n1 = (y_bin == 1).sum().float()
        n0 = (y_bin == 0).sum().float()
        n  = n0 + n1 + eps
        return float(f1_1), float(f1_0), float(n1*f1_1/n + n0*f1_0/n), float(0.5*(f1_1+f1_0))

    def _brier_mean(self, y_true, p1) -> float:
        return float(((p1 - y_true.float().to(p1.device)) ** 2).mean())

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
        x_test_alpha=None,  # if set, alpha is evaluated on this (e.g. OT-repaired features)
        **_ignored,
    ):
        # Load best beta checkpoint
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

        # Alpha metrics (evaluated on x_test_alpha if provided, else x_test)
        x_alpha = x_test_alpha if x_test_alpha is not None else x_test
        with torch.no_grad():
            p_alpha = rh.p1_from_agent(alpha_model, x_alpha)
        a_f1_min, a_f1_maj, a_f1_w, a_f1_macro = self._all_f1_from_probs(y_test, p_alpha, f1_thresh)
        a_brier = self._brier_mean(y_test, p_alpha)
        a_acc   = rh.acc_from_probs(y_test, p_alpha, f1_thresh)
        a_auc   = rh.roc_auc_from_probs(y_test, p_alpha)

        # Beta metrics
        p_beta = None
        if beta is not None:
            with torch.no_grad():
                p_beta = rh.p1_from_agent(beta, x_test)
            b_f1_min, b_f1_maj, b_f1_w, b_f1_macro = self._all_f1_from_probs(y_test, p_beta, f1_thresh)
            b_brier = self._brier_mean(y_test, p_beta)
            b_acc   = rh.acc_from_probs(y_test, p_beta, f1_thresh)
            b_auc   = rh.roc_auc_from_probs(y_test, p_beta)
        else:
            b_f1_min = b_f1_maj = b_f1_w = b_f1_macro = b_brier = b_acc = b_auc = float("nan")

        # Fairness metrics
        _nan_fair = {k: float("nan") for k in (
            "dp_diff", "eo_tpr_diff", "eod_fpr_diff", "eod_max_diff", "eod_avg_diff",
            "dp_rate_g0", "dp_rate_g1", "tpr_g0", "tpr_g1", "fpr_g0", "fpr_g1",
        )}
        if a_test is None:
            print("[TestSuite] a_test not provided; fairness metrics will be NaN.")
            a_fair = b_fair = _nan_fair
        else:
            a_fair = rh.fairness_classification_metrics(a_test, y_test, p_alpha, threshold=f1_thresh)
            b_fair = rh.fairness_classification_metrics(a_test, y_test, p_beta,  threshold=f1_thresh) \
                     if p_beta is not None else _nan_fair

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
