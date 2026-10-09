# Statistical Significance & Hypothesis Testing Protocol

This document details the paired statistical hypothesis testing protocol used to evaluate the performance impact of LLM-based synthetic data augmentation (**Translation** vs. **Translation + 1646**) across nine diverse Transformer architectures for both 5-class and binary Persian sentiment analysis.

---

## 1. Why Statistical Testing Is Needed

In empirical NLP research, simple mean comparisons across architectures can be confounded by stochastic optimization or architectural variance. To ensure rigorous evaluation:
1. **Verifying Real Gains:** We demonstrate that performance gains in fine-grained sentiment analysis (5-class) are statistically significant and not artifacts of random seed variations.
2. **Contextualizing Task Divergence:** We formalize the observed "performance ceiling" in binary sentiment classification, showing why coarse-grained tasks leave little headroom for generative augmentation compared to multi-class settings.

---

## 2. Methodology & Pipeline


### 2.1 Normality Assessment (Shapiro-Wilk)
For each metric, we test whether the paired differences $\Delta = y_{\text{augmented}} - y_{\text{baseline}}$ follow a normal distribution using the **Shapiro-Wilk test** ($\alpha = 0.05$). When normality is verified, paired Student's $t$-tests are reported; otherwise, the non-parametric Wilcoxon signed-rank test is preferred.

### 2.2 Paired Hypothesis Tests & Effect Sizes
- **Paired Student's $t$-test:** Tests whether the mean difference significantly deviates from zero:
  $$t = \frac{\bar{\Delta}}{s_{\Delta} / \sqrt{N}}$$
  Effect size is quantified using **Cohen's $d$**:
  $$d = \frac{\bar{\Delta}}{s_{\Delta}}$$

- **Wilcoxon Signed-Rank Test:** A non-parametric rank-based test evaluating median differences:
  $$W^+ = \sum_{\Delta_i > 0} \text{Rank}(|\Delta_i|)$$
  Standard normal approximation $Z$ and non-parametric effect size $r$ are computed as:
  $$Z = \frac{W^+ - \mu_W}{\sigma_W}, \quad r = \frac{|Z|}{\sqrt{N}}$$

### 2.3 Multiple Comparisons Correction (Holm-Bonferroni)
To strictly control the Family-Wise Error Rate (FWER) without excessive conservatism, we apply the step-down **Holm-Bonferroni correction** with enforced monotonicity:
$$p_{(k)}^{\text{adj}} = \max\left( \min(1.0, \, p_{(k)} \times (m - k + 1)), \, p_{(k-1)}^{\text{adj}} \right)$$

---

## 3. How to Run

### Install Dependencies
```bash
pip install numpy scipy pandas
