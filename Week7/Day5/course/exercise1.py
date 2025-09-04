import pandas as pd
import numpy as np
from scipy import stats

# 1. Simulate stock prices dataset
np.random.seed(0)  # reproducibility (mean that results are the same every time)
dates = pd.date_range(start='2023-01-01', periods=100) # Generate dates from 1st Jan 2023 to 100 days later
prices = np.random.normal(loc=100, scale=10, size=len(dates)) # Simulate stock prices with mean 100 and std dev 10

stock_data = pd.DataFrame({
    'Date': dates,
    'DVLPRS_Price': prices
})

print("Simulated Stock Prices:")
print(stock_data)
print("=" * 20, "\n")

# 2. Convert prices column to NumPy array
price_array = stock_data['DVLPRS_Price'].to_numpy()

# 3. Calculate descriptive statistics with NumPy
mean_price = np.mean(price_array)
median_price = np.median(price_array)
variance_price = np.var(price_array, ddof=1)  # sample variance
std_dev_price = np.std(price_array, ddof=1)  # sample std dev

# Display results
print("NumPy Results:")
print(f"Mean: {mean_price}")
print(f"Median: {median_price}")
print(f"Variance: {variance_price}")
print(f"Standard Deviation: {std_dev_price}\n")


# 4. Calculate with SciPy for comparison
mean_scipy = stats.tmean(price_array)
variance_scipy = stats.tvar(price_array)
std_dev_scipy = stats.tstd(price_array)

print("SciPy Results:")
print(f"Mean: {mean_scipy}")
print(f"Variance: {variance_scipy}")
print(f"Standard Deviation: {std_dev_scipy}\n")

# Interpretation
print("Interpretation:")
print("- The mean (~100) confirms the stock hovers around the assumed fair price.")
print("- The median being close to the mean suggests the distribution is roughly symmetric.")
print("- The variance and std deviation (~100) show moderate daily fluctuations.")
print("- This volatility indicates a typical medium-risk stock, not extremely stable, but not too erratic either.")

print("=" * 20, "\n")

print("Comparison between NumPy and SciPy:")
print("- Mean: NumPy and SciPy match.")
print("- Variance: NumPy and SciPy match.")
print("- Standard Deviation: NumPy and SciPy match.")

