import numpy as np
from scipy.stats import skew, kurtosis

# ===========================
# 1. Sampling and Shuffling
# ===========================
# Create a simple dataset from 1 to 100
data = np.arange(1, 101)

# Random sampling (10 values)
sample = np.random.choice(data, size=10, replace=False)

# Shuffle the sample
shuffled_sample = np.copy(sample)
np.random.shuffle(shuffled_sample)

# ===========================
# 2. Descriptive Statistics
# ===========================
# Generate a random dataset (normal distribution)
random_data = np.random.randn(1000) * 10 + 50  # Mean=50, SD=10

mean_val = np.mean(random_data)
median_val = np.median(random_data)
variance_val = np.var(random_data)
skewness_val = skew(random_data)
kurtosis_val = kurtosis(random_data)

# ===========================
# Results
# ===========================
print("=== Sampling & Shuffling ===")
print("Sample (10 values)       :", sample)
print("Shuffled Sample          :", shuffled_sample)

print("\n=== Descriptive Statistics ===")
print(f"Mean      : {mean_val:.4f}") # "Mean" means "average"
print(f"Median    : {median_val:.4f}") # "Median" means "middle value" (diff from Mean: Median is less affected by outliers)
print(f"Variance  : {variance_val:.4f}") # "Variance" means "standard deviation squared"
print(f"Skewness  : {skewness_val:.4f}") # "Skewness" means "asymmetry of the distribution", which shows how the data is skewed
print(f"Kurtosis  : {kurtosis_val:.4f}") # "Kurtosis" means "the shape of the distribution", which shows how the data is peaked
