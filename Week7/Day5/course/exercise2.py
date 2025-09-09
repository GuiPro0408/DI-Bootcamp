"""
Iris Petal Length T-Test & Visualization
---------------------------------------
- Loads Iris dataset
- Compares Setosa vs. Versicolor petal lengths using independent t-test (two-sample)
- Visualizes distributions (overlay hist + KDE) with Matplotlib (no seaborn/colors)

Notes
- This script intentionally avoids seaborn and explicit color choices.
- One chart is produced (no subplots), complying with plotting constraints.
"""

from sklearn.datasets import load_iris
from scipy.stats import ttest_ind, gaussian_kde, t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load & prepare data
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

setosa = iris_df[iris_df['species'] == 'setosa']['petal length (cm)'].to_numpy()
versicolor = iris_df[iris_df['species'] == 'versicolor']['petal length (cm)'].to_numpy()

# 2) Independent two-sample t-test (Welch by default = equal_var=False recommended)
# The original snippet used default (equal_var=True). Welch is more robust; show both.
t_stat_pooled, p_val_pooled = ttest_ind(setosa, versicolor, equal_var=True)
t_stat_welch, p_val_welch = ttest_ind(setosa, versicolor, equal_var=False)

# Calculate degrees of freedom for pooled and Welch
df_pooled = (len(setosa) + len(versicolor)) - 2
# Welch-Satterthwaite equation for degrees of freedom
s1 = setosa.var(ddof=1)
s2 = versicolor.var(ddof=1)
n1 = len(setosa)
n2 = len(versicolor)
df_welch = (s1 / n1 + s2 / n2) ** 2 / ((s1 ** 2) / ((n1 ** 2) * (n1 - 1)) + (s2 ** 2) / ((n2 ** 2) * (n2 - 1)))

# Critical value for two-tailed test, alpha=0.05
alpha = 0.05
crit_pooled = t.ppf(1 - alpha / 2, df_pooled)
crit_welch = t.ppf(1 - alpha / 2, df_welch)

print(f"Pooled-variance t-test:    T = {t_stat_pooled:.4f},  p = {p_val_pooled:.3e}, df = {df_pooled}")
print(f"Welch's t-test (robust):  T = {t_stat_welch:.4f},  p = {p_val_welch:.3e}, df = {df_welch:.2f}")
print(f"Critical value (pooled, α=0.05): ±{crit_pooled:.4f}")
print(f"Critical value (Welch, α=0.05): ±{crit_welch:.4f}")

if abs(t_stat_pooled) > crit_pooled:
    print("Pooled: Reject H0 (means differ)")
else:
    print("Pooled: Fail to reject H0 (means similar)")

if abs(t_stat_welch) > crit_welch:
    print("Welch: Reject H0 (means differ)")
else:
    print("Welch: Fail to reject H0 (means similar)")

# 3) Visualization — overlay hist + KDE (no explicit colors)
fig = plt.figure()

# Histogram overlay (density normalized)
plt.hist(setosa, bins=12, alpha=0.5, density=True, label='Iris setosa')
plt.hist(versicolor, bins=12, alpha=0.5, density=True, label='Iris versicolor')

# KDE curves (optional, improves visual intuition)
for series, lab in [(setosa, 'Setosa KDE'), (versicolor, 'Versicolor KDE')]:
    kde = gaussian_kde(series)
    xs = np.linspace(
        min(series.min(), setosa.min(), versicolor.min()) - 0.2,
        max(series.max(), setosa.max(), versicolor.max()) + 0.2,
        200
    )
    plt.plot(xs, kde(xs), linewidth=2, label=lab)

plt.title(
    f"Iris Petal Length Comparison\nT (pooled): {t_stat_pooled:.2f}, p: {p_val_pooled:.2e} | "
    f"T (Welch): {t_stat_welch:.2f}, p: {p_val_welch:.2e}"
)
plt.xlabel("Petal length (cm)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

# Uncomment to save the figure alongside analysis if running as a script
# plt.savefig("iris_ttest_plot.png", dpi=160)

plt.show()
