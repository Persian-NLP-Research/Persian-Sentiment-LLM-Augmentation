# Statistical Significance & Hypothesis Testing Protocol

This document details the paired statistical hypothesis testing protocol used to evaluate the performance impact of LLM-based synthetic data augmentation (**Translation** vs. **Translation + 1646**) across nine diverse Transformer architectures for both 5-class and binary Persian sentiment analysis.

---

## 1. Why Statistical Testing Is Needed

In empirical NLP research, simple mean comparisons across architectures can be confounded by stochastic optimization or architectural variance. To ensure rigorous evaluation:
1. **Verifying Real Gains:** We demonstrate that performance gains in fine-grained sentiment analysis (5-class) are statistically significant and not artifacts of random seed variations.
2. **Contextualizing Task Divergence:** We formalize the observed "performance ceiling" in binary sentiment classification, showing why coarse-grained tasks leave little headroom for generative augmentation compared to multi-class settings.

---

## 2. Methodology & Pipeline


## 2.1 Normality Assessment

For each metric, we test whether the paired differences

$$
\Delta_i = y_i^{\text{aug}} - y_i^{\text{base}}, \qquad i = 1, \dots, N, \; N = 9
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
W^{+} = \sum_{\Delta_i > 0} \mathrm{Rank}\!\left(|\Delta_i|\right),
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
  \min\left(1, \, (m - k + 1)\, p_{(k)}\right)
  \, p_{(k-1)}^{\text{adj}}
\right),
\qquad
p_{(0)}^{\text{adj}} = 0,
$$

for $k = 1, \dots, m$.

## 3. How to Run

### Install Dependencies
```bash
pip install numpy scipy pandas
