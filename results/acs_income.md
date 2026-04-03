# ACS Income Results

**Dataset:** ACS PUMS 2018 California (Ding et al. NeurIPS 2021)  
**Task:** Income > $50K  
**Protected attribute:** sex (female = disadvantaged, a=1, minority_id=0)  
**Config:** bias_pct=0.04, real_data_size=3000, pca=10  
**Status:** Active — competitive results obtained, regime assessment ongoing.

---

## Dataset Properties

| bias_pct | DA+ (before cap) | Notes |
|----------|-----------------|-------|
| 0.04 | ~1500 | Not in severe scarcity regime; reweighting baselines work well |
| — | — | Val/test positives: 6,427 / 6,327 (75× more than COMPAS) |

Alpha-EO at bias_pct=0.04: ~0.729 (female TPR collapses to ~0.040 under bias).

The high DA+ count means this dataset does not satisfy the scarcity regime definition (DA+≈43).
Whether competitive RL results here support a "scale robustness" claim or complicate the
scarcity narrative is under assessment.

**Reviewer framing (if included):** ACS is the community-recommended successor to UCI Adult
(Ding et al. NeurIPS 2021, "Retiring Adult"). Using alongside census_income demonstrates
scale robustness (4× data). Both use sex + income — isolates effect of dataset scale and
positive-class rate.

---

## RL Runs

| Spec | Episodes | Seeds | Status | Results |
|------|----------|-------|--------|---------|
| `acs_income_rl_200ep_1s` | 200 | [42] | smoke | — |
| `acs_income_rl_800ep_5s` | 800 | [0,2,3,5,42] | submitted to DRAC | pending |
| `acs_income_rl_1500ep_5s` | 1500 | [0,2,3,5,42] | submitted to DRAC | pending |

---

## Baseline Runs

| Spec | Method | Seeds | Status |
|------|--------|-------|--------|
| `acs_income_groupdro_5s` | GroupDRO | 5 | submitted to DRAC |
| `acs_income_otr_5s` | OT Repair | 5 | submitted to DRAC |
| `acs_income_smote_5s` | SMOTE | 5 | submitted to DRAC |
| `acs_income_flb_5s` | FLB | 5 | submitted to DRAC |

---

## Results

*Pending DRAC runs. Update this table when results arrive.*

| Method | Seeds | EO ↓ | F1w ↑ | AUC ↑ | Spec |
|--------|-------|------|-------|-------|------|
| Alpha (ERM) | — | — | — | — | — |
| GroupDRO | 5 | pending | | | `acs_income_groupdro_5s` |
| OT Repair | 5 | pending | | | `acs_income_otr_5s` |
| FLB | 5 | pending | | | `acs_income_flb_5s` |
| SMOTE | 5 | pending | | | `acs_income_smote_5s` |
| RL (v18, ep800) | 5 | pending | | | `acs_income_rl_800ep_5s` |
| RL (v18, ep1500) | 5 | pending | | | `acs_income_rl_1500ep_5s` |

---

## Notes

- Key question when results arrive: do reweighting baselines (GroupDRO, FLB) achieve low EO
  as expected given the high DA+? If yes, this dataset demonstrates regime boundary, not
  framework generality.
- If RL is competitive despite reweighting working, it may support a broader contribution
  claim — but requires honest framing about the DA+ level.
