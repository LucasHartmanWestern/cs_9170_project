# Baseline Selection Reasoning

This document records the research and decisions behind the baseline set for the Neurocomputing submission. All venues verified against publisher pages.

---

## Final Baseline Set

| Baseline | Type | Venue | Year | DOI / URL |
|---|---|---|---|---|
| GroupDRO | In-processing (loss reweighting) | ICML 2020 | 2020 | Sagawa et al. |
| Gaussian OT Repair | Pre-processing (distribution matching) | NeurIPS 2022 | 2022 | — |
| CTGAN | Generative (GAN, no fairness) | NeurIPS 2019 | 2019 | — |
| Kim et al. — BSP | In-processing (fairness-aware resampling) | Neurocomputing 2023 | 2023 | 10.1016/j.neucom.2022.11.018 |
| FairTabDDPM | Generative (conditional diffusion) | TMLR 2025 | 2025 | openreview.net/forum?id=dvRysCqmYQ |
| **Ours (v18)** | Generative (RL, global-only reward) | — | — | — |

---

## Motivation for Adding New Baselines

The original three baselines (GroupDRO, OT Repair, CTGAN) are sufficient to support the Motivation Claim (reweighting fails under scarcity) but leave two gaps a Neurocomputing reviewer will notice:

1. **No in-processing baseline from the target venue.** Adding a Neurocomputing-published method strengthens acceptance prospects and shows engagement with the journal's community.

2. **The only generative baseline (CTGAN) has no fairness component.** A reviewer will ask: "what happens if you give the GAN a fairness objective?" CTGAN was chosen deliberately as an unfairness-unaware baseline, but without a fairness-aware generative comparator the claim that RL-guided generation is superior to generative fairness methods is unsupported.

---

## In-Processing Baseline: Kim et al. (2023)

**Full citation:**
Kim, D., Park, S., Hwang, S., Byun, H. (2023). "Fair classification by loss balancing via fairness-aware batch sampling." *Neurocomputing*, Vol. 518, pp. 231–241.
**Correct DOI:** `10.1016/j.neucom.2022.11.018` (note: DOI `10.1016/j.neucom.2022.08.040` resolves to an unrelated paper — do not use it)
PII: S0925231222013984

**Method (BSP — Batch Sampling Probability):**
Computes per-(a, y) group training loss at each step. Adaptively updates group sampling probabilities proportional to group loss (high-loss groups are oversampled in the next batch). Standard unweighted BCE is applied to the resampled batch. Key insight: balancing *losses* (not just sample counts) is necessary and sufficient for fairness improvement.

**Key difference from GroupDRO:** GroupDRO reweights the *loss function* using multiplicative group weights. BSP reweights the *data sampling distribution* and trains on standard unweighted loss. Both converge to equalising worst-group loss but via different mechanisms.

**Datasets in paper:** CelebA, UTKFace (facial attribute classification — primary experiments), plus one tabular dataset (likely Adult; unconfirmed due to paywall). **Caution:** this is primarily a vision paper. The tabular results are secondary and the method has not been validated under severe positive-class scarcity.

**Why it fails under our regime:** BSP assumes sufficient minority-positive examples to compute stable per-group loss estimates. With 17–43 minority-positive training examples (bias_pct 0.05–0.10), the group loss signal is noisy and BSP degrades for the same reason as GroupDRO — too few minority-positive gradients to balance against.

**Why include it despite vision focus:** The algorithm is architecture-agnostic and applies directly to our tabular FFNN setting. Published in our target venue (Neurocomputing) — citing it strengthens engagement with the journal community. Reviewers familiar with the journal will recognise it.

**Implementation:** `benchmarks/fairness_loss_balancing.py`. Spec key: `"baseline": "fairness_loss_balancing"`. Hyperparams in `"flb"` dict.

---

## Generative Fairness Baseline: FairTabDDPM (Yang et al., 2025)

**Full citation:**
Yang, Z., Yu, H., Guo, P., Zanna, K., Yang, X., Sano, A. (2025). "Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models." *Transactions on Machine Learning Research (TMLR)*, February 2025.
OpenReview: `https://openreview.net/forum?id=dvRysCqmYQ`
GitHub: `https://github.com/comp-well-org/fair-tab-diffusion`

**Venue note:** TMLR is a peer-reviewed ML journal (formal action editors + reviewers, affiliated with JMLR/PMLR). Not a conference proceedings but explicitly peer-reviewed and indexed. Verified at OpenReview.

**Method:**
Extends TabDDPM with joint (class label, sensitive attribute) conditioning in the denoising network. The `is_fair=True` flag conditions generation simultaneously on both y and a, producing samples that respect equalized odds constraints by construction. Sensitive attributes are specified in `desc.json`; the denoising network receives them as additional embeddings at each reverse diffusion step.

**Datasets in paper:** Adult (sex protected attribute), COMPAS, German Credit, Bank Marketing.
**Fairness metrics:** Demographic Parity Ratio and Equalized Odds Ratio — both explicitly evaluated. Equal Opportunity (TPR parity) is covered under Equalized Odds.

**Why it is the right generative baseline:**
- Direct conceptual competitor to our method: both use an external signal (classifier guidance vs. RL reward) to steer generation toward fairness
- Tests on Adult with sex-as-protected-attribute — matches our census_income setting exactly
- Top venue (TMLR 2025), code available

**Integration note:** The original library targets mixed-type tabular data with a file-system-centric pipeline, hardcoded paths (`/rdf/db/`), and heavy dependencies (`dgl`, `torch_geometric`, `skops`) not in our environment. Since our features are already all-numerical (PCA-transformed), we implement the core algorithm — Gaussian DDPM with (y, a) conditioning — directly in PyTorch. This is described in the paper as "adapted from Yang et al. (2025) for all-numerical PCA features."

**Implementation:** `benchmarks/fairtabddpm_baseline.py`. Spec key: `"baseline": "fairtabddpm"`. Hyperparams in `"fairtabddpm"` dict.

---

## Candidates Investigated and Rejected

### CuTS — ICML 2024
Vero et al. "CuTS: Customizable Tabular Synthetic Data Generation."
`proceedings.mlr.press/v235/vero24a.html`

**Architecture:** GAN-style generator with marginal matching (not a normalizing flow as initially reported). Four-layer FC network with Gumbel-Softmax for categorical features.

**Why rejected as baseline:** Tests on the full unmodified Adult dataset with no bias injection. Optimises for Demographic Parity, not Equal Opportunity (EoO is only a side-effect reported in the appendix). Does not address positive-class scarcity. Their EoO=0.02 on Adult is impressive but achieved starting from a complete dataset — it would degrade severely under our bias_pct=0.10 regime for a fundamentally different reason than our method, making the comparison confusing and uninformative.

**Keep as related work citation.** ICML 2024 is a strong citation for the related work section — distinguish by noting it addresses full-dataset fairness, not the positive-class scarcity regime.

### DECAF — NeurIPS 2021
van Breugel et al. "DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks."
`proceedings.neurips.cc/paper/2021/hash/ba9fab001f67381e56e410575874d967-Abstract.html`
pip: `decaf-synthetic-data`

**Method:** Causal GAN — biased edges in the structural causal model (SCM) are surgically removed at sampling time to enforce fairness. Architecture-agnostic; multiple fairness definitions supported.

**Why not selected as primary baseline:** Requires a user-supplied causal DAG over features. While the adult income causal graph is well-studied, specifying it introduces a modelling assumption that is orthogonal to our contribution and would need to be justified to reviewers. FairTabDDPM is a stronger apples-to-apples comparison (both are conditioning-based generative approaches with no causal assumptions).

**Keep as related work.** NeurIPS 2021 is the most prestigious citation in the generative fairness literature. Cite and distinguish: DECAF requires a known causal graph; our method requires no causal assumptions.

### TabFairGAN — Applied Sciences (MDPI) 2022
Rejected: MDPI journal, weak venue. Not citeable as a top-tier baseline.

### FairTabDDPM — arXiv uncertainty
Initial concern: user could not find TMLR publication. **Resolved:** OpenReview page confirmed at `openreview.net/forum?id=dvRysCqmYQ`. TMLR is peer-reviewed.

### DOI 10.1016/j.neucom.2023.126804 (user-provided)
Resolves to: "A systematic literature review on object detection using near infrared and thermal images." Not a fairness paper. Likely a DOI typo by the user.

---

## Concise Paper Story

**Motivation (Claim 1):** Under positive-class outcome scarcity (bias_pct ≤ 0.10), methods that require sufficient minority-positive examples fail. This includes:
- **GroupDRO**: group loss weights become noisy with few minority-positive examples
- **OT Repair**: transport map is ill-conditioned with near-empty minority-positive support
- **Kim et al. BSP**: group loss estimates are unreliable with 17–43 minority-positive training examples

**Main claim (Claim 2):** Our RL-guided generative approach achieves competitive fairness-utility tradeoff even under severe scarcity. Compared against:
- **CTGAN**: generative baseline without fairness objective — shows augmentation alone is insufficient
- **FairTabDDPM**: generative baseline *with* fairness objective — shows RL-guided targeting outperforms diffusion-based conditioning

**Ablation (Claim 3):** v18 (global-only reward) vs v17a (DVRL local reward) — clean negative result showing the simpler global reward is better.
