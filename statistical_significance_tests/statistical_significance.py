"""
Statistical Significance Testing Pipeline for Persian Sentiment Classification.
Compares Baseline vs. LLM-Augmented across 9 models.
Includes:
- Shapiro-Wilk Normality Audit
- Paired Student's t-test + Cohen's d
- Wilcoxon Signed-Rank Test + Effect size r
- Holm-Bonferroni Family-Wise Error Rate (FWER) step-down correction
"""

import numpy as np
import pandas as pd
from scipy import stats

DATA = {
    # The following values are for illustration purposes only. When running the code, enter the values based on the experiment performed.
        
    # 5-CLASS TASK Results
    ("5-Class", "Accuracy"): {
        "Baseline":  [72.64, 72.64, 70.16, 71.11, 69.00, 70.68, 75.30, 72.50, 72.73],
        "Augmented": [72.93, 73.07, 70.68, 71.60, 70.97, 71.45, 75.89, 74.20, 73.31]
    },
    ("5-Class", "Weighted-F1"): {
        "Baseline":  [72.92, 73.03, 70.57, 71.50, 69.32, 70.61, 75.53, 72.93, 73.19],
        "Augmented": [73.33, 73.53, 71.09, 72.06, 71.24, 71.47, 76.09, 74.49, 73.72]
    },
    ("5-Class", "Macro-F1"): {
        "Baseline":  [66.04, 64.85, 62.56, 62.15, 59.70, 59.47, 70.08, 64.22, 65.30],
        "Augmented": [65.09, 64.86, 61.86, 61.56, 63.23, 60.76, 70.08, 67.46, 66.11]
    },

    # Binary TASK Results
    ("Binary", "Accuracy"): {
        "Baseline":  [93.76, 93.88, 92.53, 92.47, 90.85, 94.48, 96.55, 94.60, 94.36],
        "Augmented": [93.85, 93.97, 92.41, 93.22, 91.09, 95.05, 96.58, 94.12, 94.66]
    },
    ("Binary", "Weighted-F1"): {
        "Baseline":  [93.83, 93.88, 92.57, 92.56, 91.00, 94.55, 96.55, 94.66, 94.40],
        "Augmented": [93.93, 94.04, 92.45, 93.30, 91.28, 95.08, 96.60, 94.20, 94.72]
    },
    ("Binary", "Macro-F1"): {
        "Baseline":  [89.50, 89.48, 87.31, 87.36, 84.76, 90.76, 94.07, 90.91, 90.42],
        "Augmented": [89.69, 89.86, 87.10, 88.62, 85.34, 91.59, 94.17, 90.18, 91.01]
    }
}

def holm_bonferroni(p_values):
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])   #Enforces monotonicity in Holm-Bonferroni step-down correction.
    m = len(p_values)
    adj = [0.0] * m
    prev = 0.0
    for rank, (original_idx, p) in enumerate(indexed):
        val = min(p * (m - rank), 1.0)
        val = max(val, prev)  # enforce monotonicity
        adj[original_idx] = val
        prev = val
    return adj

records = []
raw_wilcox_p = []
raw_ttest_p = []

for (task, metric), values in DATA.items():
    base = np.array(values["Baseline"], dtype=float)
    aug = np.array(values["Augmented"], dtype=float)
    diff = aug - base

    d_nonzero = diff[diff != 0]
    n_nonzero = len(d_nonzero)
    n_total = len(diff)

    mean_base = np.mean(base)
    mean_aug = np.mean(aug)
    mean_gain = np.mean(diff)
    wins = int(np.sum(diff > 0))

    # 1. Shapiro-Wilk test on differences
    shapiro_stat, p_shapiro = stats.shapiro(diff)
    is_normal = "Yes" if p_shapiro >= 0.05 else "No"

    # 2. Statistical Tests
    t_stat, p_ttest = stats.ttest_rel(aug, base)
    w_res = stats.wilcoxon(aug, base)
    p_wilcox = w_res.pvalue

    raw_wilcox_p.append(p_wilcox)
    raw_ttest_p.append(p_ttest)

    # 3. Correct Wilcoxon Z & Effect Size r
    ranks = stats.rankdata(np.abs(d_nonzero))
    w_plus = ranks[d_nonzero > 0].sum()
    w_minus = ranks[d_nonzero < 0].sum()

    mean_w = n_nonzero * (n_nonzero + 1) / 4.0
    std_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
    z_val = (w_plus - mean_w) / std_w
    effect_r = abs(z_val) / np.sqrt(n_nonzero)

    # 4. Cohen's d for paired samples
    cohen_d = mean_gain / np.std(diff, ddof=1)

    records.append({
        "Task": task,
        "Metric": metric,
        "Mean_Base": round(mean_base, 2),
        "Mean_Aug": round(mean_aug, 2),
        "Gain": round(mean_gain, 2),
        "Wins": f"{wins}/{n_total}",
        "Shapiro_p": round(p_shapiro, 4),
        "Is_Normal": is_normal,
        "Wilcoxon_p": round(p_wilcox, 4),
        "ttest_p": round(p_ttest, 4),
        "Z_val": round(z_val, 3),
        "Effect_r": round(effect_r, 3),
        "Cohens_d": round(cohen_d, 2)
    })

# Apply Holm to both sets of p-values
adj_p_wilcox = holm_bonferroni(raw_wilcox_p)
adj_p_ttest = holm_bonferroni(raw_ttest_p)

for i, rec in enumerate(records):
    rec["Holm_Wilcoxon_p"] = round(adj_p_wilcox[i], 4)
    rec["Holm_ttest_p"] = round(adj_p_ttest[i], 4)
    rec["Sig_Wilcoxon"] = "Yes" if adj_p_wilcox[i] < 0.05 else "No"
    rec["Sig_ttest"] = "Yes" if adj_p_ttest[i] < 0.05 else "No"

df_final = pd.DataFrame(records)
print(df_final.to_string(index=False))

output_file = "comprehensive_significance_report.csv"
df_final.to_csv(output_file, index=False)
print(f"\n Results saved'{output_file}'")
