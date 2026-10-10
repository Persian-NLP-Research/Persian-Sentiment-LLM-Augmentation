# Statistical Significance & Hypothesis Testing Protocol

This document details the paired statistical hypothesis testing protocol used to evaluate the performance impact of LLM-based synthetic data augmentation across nine diverse Transformer architectures for both 5-class and binary Persian sentiment analysis.

---

## 1. Why Statistical Testing Is Needed

In empirical NLP research, simple mean comparisons across architectures can be confounded by stochastic optimization or architectural variance. To ensure rigorous evaluation:
1. **Verifying Real Gains:** We demonstrate that performance gains in fine-grained sentiment analysis (5-class) are statistically significant and not artifacts of random seed variations.
2. **Contextualizing Task Divergence:** We formalize the observed "performance ceiling" in binary sentiment classification, showing why coarse-grained tasks leave little headroom for generative augmentation compared to multi-class settings.


Evaluating generative data augmentation across multiple architectural families introduces specific statistical considerations that govern our methodological choices:

- **Paired Design Across Identical Architectures ($N = 9$):**  
   Because both baseline and augmented training configurations are trained and evaluated across the exact same set of $9$ transformer-based architectures under controlled random seeds, performance observations are inherently dependent. A **paired testing paradigm** is statistically mandatory to account for inter-model variance, isolating the true marginal effect of generative augmentation ($\Delta_i = y_i^{\text{aug}} - y_i^{\text{base}}$) rather than inter-architecture capacity differences.

- **Normality Verification & Dual Reporting (Parametric vs. Non-Parametric):**  
   With a moderate sample size ($N = 9$), asymptotic normality cannot be assumed a priori. We explicitly perform the **Shapiro-Wilk test** on the paired differences $\Delta_i$. While the decision rule formally defers to the **Wilcoxon signed-rank test** whenever normality is violated ($\alpha < 0.05$) and the **paired Student's $t$-test** otherwise, we report both tests alongside standardized effect sizes (Cohen's $d_z$ and Wilcoxon $r$) to guarantee full empirical transparency.

- **Family-Wise Error Rate (FWER) Control via Step-Down Holm-Bonferroni:**  
   Simultaneously evaluating multiple classification metrics (Accuracy, Weighted-F1, Macro-F1) across task regimes (Binary vs. 5-Class) induces a severe risk of Type I error inflation ($\alpha$-multiplicity). Rather than employing standard Bonferroni correction, which is overly conservative and inflates Type II errors, we adopt the **Holm-Bonferroni step-down procedure** with enforced monotonicity. This rigorously controls the Family-Wise Error Rate (FWER $\le 0.05$) while preserving statistical power to detect genuine task-dependent performance gains.


---

## 2. Methodology & Pipeline


## 2.1 Normality Assessment

For each metric, we test whether the paired differences

$$
\Delta_i = y_i^{\text{aug}} - y_i^{\text{base}}, \qquad i = 1, \dots, N
$$

follow a normal distribution using the **Shapiro-Wilk test** at $\alpha = 0.05$.
Both paired Student's $t$-test and Wilcoxon signed-rank test are reported for full
transparency; the decision rule follows the Shapiro-Wilk outcome.

## 2.2 Paired Hypothesis Tests

**Paired Student's $t$-test.**

$$
t = \frac{\bar{\Delta}}{s_{\Delta} / \sqrt{N}}
$$

with Cohen's $d_z$ as the effect size:

$$
d_z = \frac{\bar{\Delta}}{s_{\Delta}}
$$

**Wilcoxon signed-rank test.** Zeros are discarded (`zero_method='wilcox'`),
leaving $N'$ non-zero pairs. Define

$$
W^{+} = \sum_{\Delta_i > 0} \mathrm{Rank}\left(|\Delta_i|\right),
\qquad
\mu_W = \frac{N'(N'+1)}{4},
\qquad
\sigma_W = \sqrt{\frac{N'(N'+1)(2N'+1)}{24}}.
$$

The standardized statistic with continuity correction is

$$
Z = \frac{W^{+} - \mu_W - 0.5\,\mathrm{sign}(W^{+} - \mu_W)}{\sigma_W},
\qquad
r = \frac{|Z|}{\sqrt{N'}}.
$$

## 2.3 Multiple Comparisons Correction

Let

$$
p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}
$$

be the ordered raw $p$-values ($m = 6$). The Holm-Bonferroni adjusted $p$-values
are computed recursively:

$$
p_{(k)}^{\text{adj}} =
\max\left(
  \min\left(1 \, (m - k + 1)\, p_{(k)}\right)
  \, p_{(k-1)}^{\text{adj}}
\right),
\qquad
p_{(0)}^{\text{adj}} = 0,
$$

for $k = 1, \dots, m$.

## 3. How to Run

### 3.1 Repository Layout

```
statistical_significance_tests/
├── README.md                                         # This Document!
├── requirements.txt
├── statistical_significance.py                       # Generic, reusable statistical script (Python)
├── Original_vs_Original(+1000).ipynb                 
├── Original_vs_Original(+1146).ipynb                 
├── Original_vs_Original(+1646).ipynb                 
├── Balanced_vs_Balanced(+1646).ipynb                 
└── Translation_vs_Translation(+1646).ipynb          
```

- **`statistical_significance.py`** : The generic implementation of all statistical procedures
  (Shapiro-Wilk normality test, paired Student's *t*-test, Wilcoxon signed-rank
  test, Cohen's *d*<sub>z</sub>, effect size *r*, and the Holm-Bonferroni
  step-down correction). To reuse it, simply replace the `DATA` dictionary with
  the values of your own experiment and run the script.

- **The five notebooks** :
  [`OriginalVsOriginal1000.ipynb`](https://github.com/Persian-NLP-Research/Persian-Sentiment-LLM-Augmentation/blob/main/statistical_significance_tests/Original_vs_Original(%2B1000).ipynb),
  [`OriginalVsOriginal1146.ipynb`](OriginalVsOriginal1146.ipynb),
  [`OriginalVsOriginal1646.ipynb`](OriginalVsOriginal1646.ipynb),
  [`BalancedVsBalanced1646.ipynb`](BalancedVsBalanced1646.ipynb), and
  [`TranslationVsTranslation1646.ipynb`](TranslationVsTranslation1646.ipynb).
  Each corresponds to one experimental scenario. They share **identical
  statistical logic** and differ **only in the input data**. Each notebook is designed to be **copied and pasted directly
  into Google Colab**.

- **`requirements.txt`** : pinned list of Python dependencies
  (`numpy`, `pandas`, `scipy`).

  
