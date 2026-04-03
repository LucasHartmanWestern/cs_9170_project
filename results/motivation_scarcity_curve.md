# Motivation: Scarcity Curve (Census Income)

**Question:** Do reweighting baselines degrade as DA+ decreases?  
**Dataset:** census_income, 3 seeds [42,0,1], v16 config  
**Source:** v16 bias sweep (training_runs/, 2026-03 experiments)

---

## GroupDRO vs RL across bias levels

| bias_pct | DA+ (approx) | GroupDRO EO | RL (v16) EO |
|----------|-------------|-------------|-------------|
| unbiased | ~300+ | 0.029 ± 0.026 | 0.083 (v14) |
| 0.15 | ~77 | 0.018 ± 0.007 | — |
| 0.10 | 43 | 0.100 ± 0.038 | 0.084 ± 0.043 |
| 0.05 | 17 | **0.247 ± 0.120** | 0.097 ± 0.067 |

GroupDRO collapses at DA+=17 (EO=0.247, 8× worse than unbiased) and is still meaningfully
impaired at DA+=43 (EO=0.100 vs 0.029 unbiased). RL remains stable across scarcity levels.

---

## Full baseline degradation (census, bias=0.10, DA+=43)

From v16 3-seed runs, this was the original motivation table:

| Method | bias=0.05 EO | bias=0.10 EO | bias=0.15 EO | unbiased EO |
|--------|-------------|-------------|-------------|-------------|
| GroupDRO | **0.247** ± 0.120 | **0.100** ± 0.038 | 0.018 ± 0.007 | 0.029 ± 0.026 |
| OT Repair | consistent utility cost across all levels (~F1w 0.757–0.788) | | | |
| RL (v16) | 0.097 ± 0.067 | 0.084 ± 0.043 | — | 0.083 |

---

## Status / Paper Decision

**2026-04-03:** Whether to use this as a paper motivation figure is under consideration.
The degradation effect is real and well-supported by census results. The concern is whether
it is compelling enough given that FLB (a reweighting method) also works reasonably well
on census at bias=0.10 — it's GroupDRO specifically that fails, not all baselines.

**Strongest version of the claim:** Focus on COMPAS sex, where GroupDRO (0.156), FLB (0.118),
CTGAN (0.158), and SMOTE (0.507) all worsen EO vs alpha (0.070). This is a cleaner,
more dramatic demonstration of baseline failure under scarcity than the census degradation curve.

**If including the curve:** Frame as "GroupDRO degrades monotonically with DA+" using the
census bias sweep. Do not generalize to all reweighting methods — FLB is a counter-example.

---

## Additional scarcity smokes (census, 2 seeds [0,2], 2026-03-28)

Ran to check whether initial episode return drops at less severe scarcity, validating
that the high early returns at DA+=43 are a scarcity-specific phenomenon:

| bias_pct | DA+ (approx) | Expected initial return |
|----------|-------------|------------------------|
| 0.05 | 17 | higher (more scarce) |
| 0.10 | 43 | 0.93 (measured) |
| 0.15 | ~77 | lower (less scarce) |
| 0.20 | ~96 | lower (less scarce) |

Results from these smokes were not fully recorded — check `/tmp/scarcity_monitor.log`
on the machine where they ran, or rerun if needed.
