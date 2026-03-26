# Fairness-Aware Generative Baseline Comparison

Comparing candidate methods for the fairness-aware synthetic data generator baseline slot.
Our setting: binary classification, sex-protected attribute, Equal Opportunity (TPR parity), severe positive-class scarcity (bias_pct ≤ 0.10).

---

## Comparison Table

| | **FairTabDDPM** | **FairGAN** | **CuTS** | **DECAF** | **TabFairGAN** | **FLDGMs** |
|---|---|---|---|---|---|---|
| **Venue** | TMLR 2025 (OpenReview — verify before citing) | IEEE Big Data 2018 | ICML 2024 | NeurIPS 2021 | Applied Sciences / MDPI 2022 | ECAI 2023 |
| **Venue strength** | ✅ Peer-reviewed ML journal | ⚠️ Older IEEE conference | ✅ Top-tier | ✅ Top-tier | ❌ Weak (MDPI) | ✅ Decent (23% accept rate) |
| **Architecture** | Diffusion (DDPM) | GAN | GAN + marginal matching | Causal GAN | GAN | VAE/GAN hybrid (modular backend) |
| **Primary fairness metric** | DP ratio + EO ratio | DP | DP (EO in appendix only) | DP, EO, counterfactual fairness | DP, EO | DP, EO |
| **EO (TPR parity) explicitly tested** | ✅ Yes | ⚠️ Partial | ⚠️ Appendix only | ✅ Yes | ✅ Yes | ✅ Yes |
| **Tested on tabular data** | ✅ Adult, COMPAS, German Credit | ✅ Adult | ✅ Adult | ✅ Adult, synthetic | ✅ Adult, German Credit | ✅ Adult |
| **Tested under positive-class scarcity** | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **Requires causal graph** | ❌ No | ❌ No | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **Code available** | ✅ GitHub | ✅ GitHub (older) | ✅ GitHub | ✅ pip (`decaf-synthetic-data`) | ✅ pip (`tabfairgan`) | ✅ GitHub (Docker) |
| **Integration effort** | Medium (our from-scratch impl. already done) | Low | Medium | Low-medium | Low | Medium-high |
| **Extra dependencies** | Heavy (dgl, pyg) if using orig. library — none for our impl. | Minimal | Minimal | Minimal | Minimal | Docker |
| **Recency** | ✅ 2025 | ❌ 2018 | ✅ 2024 | ✅ 2021 | ⚠️ 2022 | ✅ 2023 |

---

## Pros & Cons

### FairTabDDPM — Yang et al. (TMLR 2025)
**Pros:**
- Most recent (2025), state-of-the-art diffusion approach
- Explicitly targets EO on Adult with sex protected attribute — exact match to our setting
- Our from-scratch implementation already written and ready
- Joint (y, a) conditioning is a principled and direct conceptual competitor to our RL reward

**Cons:**
- TMLR peer-review status needs supervisor/personal verification before citing
- Implementation is our own adaptation (not the original library) — reviewers could ask why
- Diffusion models are slow to train (1000 epochs) vs. GAN alternatives

---

### FairGAN — Xu et al. (IEEE Big Data 2018)
**Pros:**
- Simple and well-understood architecture
- Low integration effort, code available
- Original GAN-based fairness method — historically important

**Cons:**
- 2018 — old for a 2025 submission; reviewers will ask why a more recent method wasn't used
- IEEE Big Data is not a prestigious venue relative to NeurIPS/ICML/TMLR
- DP-focused; EO only partially evaluated

**Verdict: Too old, too weak a venue. Better as a related work citation than an implemented baseline.**

---

### CuTS — Vero et al. (ICML 2024)
**Pros:**
- Strongest conference venue (ICML 2024)
- No causal graph required, declarative constraint specification
- Achieves excellent EoO on Adult (0.02) in their setting

**Cons:**
- Tested on full unbiased dataset — not designed for positive-class scarcity
- Primarily optimises DP, not EO directly
- Would degrade under our bias_pct regime for fundamentally different reasons — makes comparison confusing and potentially misleading for reviewers
- Architecture is GAN-style but the paper doesn't present it as a "fairness GAN" — framing mismatch

**Verdict: Wrong problem setting. Better as a related work citation with the note that it targets full-dataset fairness, not scarcity.**

---

### DECAF — van Breugel et al. (NeurIPS 2021)
**Pros:**
- Strongest confirmed venue (NeurIPS 2021)
- Multiple fairness notions supported
- pip-installable, low integration effort
- Well-cited (strong signal it is the reference causal fairness generator)

**Cons:**
- Requires a user-specified causal DAG — introduces an assumption orthogonal to our contribution that reviewers will scrutinise
- Causal graph for census/credit must be justified and may be contested
- Performance on CuTS table shows DECAF has poor accuracy (66.8% vs 82-84% for others) — may make the comparison look unfavourable without careful framing

**Verdict: Best venue, but causal graph requirement is a significant burden. Worth including as related work. Could be implemented if supervisor prefers top-venue comparison.**

---

### TabFairGAN — Rajabi & Garibay (Applied Sciences / MDPI 2022)
**Pros:**
- Simple GAN, pip-installable, designed specifically for tabular fairness
- DP and EO both evaluated

**Cons:**
- MDPI journal — widely regarded as low-tier / predatory-adjacent
- User has already rejected this venue

**Verdict: Ruled out. Venue is unacceptable.**

---

### FLDGMs — Ramachandranpillai et al. (ECAI 2023)
**Pros:**
- Decent peer-reviewed venue (ECAI 2023, 23% acceptance rate)
- Modular design separates fairness representation from generation
- Evaluates DP and EO on Adult
- More recent than DECAF

**Cons:**
- Less prestigious than NeurIPS/ICML/TMLR
- Docker-only deployment adds friction
- Fewer citations than DECAF/CuTS — less established

**Verdict: Backup option if FairTabDDPM TMLR status is not confirmed. ECAI is solid but not as strong as TMLR.**

---

## Recommendation

**If TMLR is confirmed peer-reviewed:** → FairTabDDPM (already implemented, best fit for our setting, most recent)

**If TMLR cannot be confirmed:** → FLDGMs (ECAI 2023) as the fallback generative baseline, with DECAF cited as the strongest related work

**Regardless:** Cite DECAF (NeurIPS 2021) and CuTS (ICML 2024) in related work — they are the two most prestigious adjacent citations and distinguish your problem framing (positive-class scarcity) from theirs (full-dataset fairness).
