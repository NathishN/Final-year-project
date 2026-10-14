# Phase 5.6 — Forensic Statistical Reconciliation & Exact Permutation Audit

**Audit Timestamp:** 2026-10-14 14:09:12  

**Software Stack:** Python 3.12.4, SciPy 1.17.1, NumPy 2.4.0, PyTorch 2.5.1+cu121  

**Evaluation Scope:** Forensic recomputation of paired parametric and non-parametric tests, complete resolution of the Wilcoxon tied-rank permutation distribution, and Holm–Bonferroni multiplicity control ($m=3$).

---

## 1. Forensic Resolution of the Wilcoxon Signed-Rank Test with Tied Differences


### A. Exact Ranks and Statistic Definition

In the historical experiment across $n=30$ paired random seeds, the paired differences $\Delta \text{QWK} = \text{QWK}_B - \text{QWK}_A$ were extracted directly from per-seed prediction tensors.

Upon ranking the 30 absolute differences $|\Delta \text{QWK}|$:

- **Seed 3:** $\text{QWK}_A = 0.8338, \text{QWK}_B = 0.8601 \implies \Delta = +0.0263$

- **Seed 73:** $\text{QWK}_A = 0.8626, \text{QWK}_B = 0.8363 \implies \Delta = -0.0263$


Both seeds possess an identical absolute difference of exactly **$0.026300$**. These two samples span ranks 13 and 14.

Under standard statistical tie-averaging (`scipy.stats.rankdata(..., method='average')`), each receives rank **$13.5$**:

- Positive rank sum: $W^+ = 58.0 + 13.5 = \mathbf{71.5}$ ($9$ positive differences).

- Negative rank sum: $W^- = 380.0 + 13.5 = \mathbf{393.5}$ ($21$ negative differences).

- Total rank sum: $W^+ + W^- = 465.0 = \frac{30 \times 31}{2}$.

- **Wilcoxon Statistic Convention:** We explicitly define $W = \min(W^+, W^-) = \mathbf{71.5}$.


### B. Discovery: Why SciPy `method='exact'` Yielded $0.000555$ while True Permutation Yields $0.000514$

A rigorous algorithmic inspection reveals why different p-values have appeared:

1. **True Conditional Permutation Test (Accounting for Tied Ranks):**

   - Conditioned on the actual observed rank vector containing the fractional tie ($13.5, 13.5$), there are $2^{30} = 1,073,741,824$ equally likely sign assignments under $H_0$.

   - Using dynamic programming to enumerate the exact distribution of $W^+$ under this specific rank vector, exactly **$276,128$** combinations produce $W^+ \le 71.5$, and by symmetry **$276,128$** produce $W^+ \ge 393.5$.

   - **True Two-Sided Exact Conditional Permutation $p$-Value:**

     $$p = \frac{276,128 + 276,128}{1,073,741,824} = \frac{552,256}{1,073,741,824} = \mathbf{0.000514328 \dots \approx 0.000514}$$

2. **SciPy's `method='exact'` (No-Ties Theoretical Distribution):**

   - SciPy 1.17's internal exact Wilcoxon lookup assumes strictly integer ranks $1, 2, \dots, 30$ (ignoring the tie structure) and evaluates $W = \lceil 71.5 \rceil = 72$ against the standard integer distribution.

   - In the standard integer no-ties distribution, the number of sign combinations with sum $\le 72$ is $297,870$, yielding:

     $$p_{\text{scipy}} = \frac{2 \times 297,870}{2^{30}} = \mathbf{0.000554826 \dots \approx 0.000555}$$

3. **Asymptotic Normal Approximation:**

   - Uses the normal distribution with variance corrected for one tie of size 2 ($t_1=2$):

     $$\sigma^2 = \frac{30(31)(61)}{24} - \frac{2^3-2}{48} = 2363.75 - 0.125 = 2363.625$$

     $$z = \frac{71.5 - 232.5}{\sqrt{2363.625}} = \frac{-161.0}{48.6171} = -3.31159 \implies p = 2 \times \Phi(-3.3116) = \mathbf{0.000928}$$

> **Definitive Scientific Resolution:**  

> - The mathematically exact p-value conditioned on the observed data is **$p = 0.000514$**.

> - The value $0.000555$ is SciPy's integer-rank lookup.

> - The value $0.000928$ is the asymptotic normal approximation.

> All three values confirm significant performance degradation ($p < 0.001$). Both the exact permutation p-value ($0.000514$) and asymptotic p-value ($0.000928$) are documented side-by-side.


---

## 2. Authoritative Master Statistical Table


### Sample Size Nomenclature:

- **$N_{\text{test}}$:** Total evaluated test images ($N=12,833$ in Historical V2, $N=12,980$ in 100% Group Session Holdout).

- **$n_{\text{seeds}} = 30$:** Number of paired model-training seeds, yielding degrees of freedom $\text{df} = 29$ for paired statistical tests.

- **Statistic Convention:** $W = \min(W^+, W^-)$ is reported, with $W^+$ and $W^-$ listed explicitly.


| Protocol | Contrast | $N_{\text{test}}$ | $n_{\text{seeds}}$ | Mean Diff ($\Delta$ QWK) | 95% Conf. Int. | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | $W = \min(W^+, W^-)$ | $W^+$ / $W^-$ | Exact Perm $p_w$ | Holm $p_{w,\text{adj}}$ | Asymptotic $p_w$ | Pairs (+/-) |

|:---|:---:|---:|:---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|

| **Historical V2** | **B vs A** | 12,833 | 30 | **-0.0360** | [-0.0537, -0.0183] | -0.7593 | -4.1590 | 2.593711e-04 | **2.593711e-04** | **71.5** | 71.5 / 393.5 | **5.143285e-04** | **5.143285e-04** | 9.276733e-04 | 9 / 21 |

| **Historical V2** | **C vs A** | 12,833 | 30 | **-0.0700** | [-0.0902, -0.0498] | -1.2947 | -7.0911 | 8.408533e-08 | **1.681707e-07** | **16.0** | 16.0 / 449.0 | **3.147870e-07** | **6.295741e-07** | 8.466082e-06 | 2 / 28 |

| **Historical V2** | **D vs A** | 12,833 | 30 | **-0.0848** | [-0.1008, -0.0687] | -1.9713 | -10.7971 | 1.126546e-11 | **3.379638e-11** | **0.0** | 0.0 / 465.0 | **1.862645e-09** | **5.587935e-09** | 1.734398e-06 | 0 / 30 |

| **Session Holdout** | **B vs A** | 12,980 | 30 | **-0.0262** | [-0.0456, -0.0067] | -0.5019 | -2.7490 | 1.018018e-02 | **1.018018e-02** | **125.0** | 125.0 / 340.0 | **2.588827e-02** | **2.588827e-02** | 2.702511e-02 | 10 / 20 |

| **Session Holdout** | **C vs A** | 12,980 | 30 | **-0.0525** | [-0.0707, -0.0342] | -1.0720 | -5.8715 | 2.267368e-06 | **4.534735e-06** | **29.0** | 29.0 / 436.0 | **3.239140e-06** | **6.478280e-06** | 2.843424e-05 | 4 / 26 |

| **Session Holdout** | **D vs A** | 12,980 | 30 | **-0.0874** | [-0.1081, -0.0667] | -1.5744 | -8.6231 | 1.697994e-09 | **5.093981e-09** | **9.0** | 9.0 / 456.0 | **6.146729e-08** | **1.844019e-07** | 4.285686e-06 | 2 / 28 |


---

## 3. Scope of Statistical Conclusions (Scientific Qualification)

1. **Bounded Generalization:** The 30 random seeds represent repeated model initializations and mini-batch sample orderings evaluated on the defined test splits. The statistical significance ($p < 0.05$ across all tests under Holm correction) establishes that auxiliary conditioning consistently degrades ordinal grading **across the evaluated training seeds on these specific protocols**. This is an empirical demonstration on the multi-fruit benchmark, not an assertion of universal invariance across all agricultural vision tasks.

2. **Protocol Persistence:** The negative effect of species conditioning persisted when transitioning from the historical split (with partial session leakage) to the strict session-held-out protocol (with zero session leakage), demonstrating that the observed degradation was not an artifact of session clustering.

