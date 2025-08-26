"""
Synthetic Retail Sales Data Generator
-------------------------------------
This script generates a realistic retail sales dataset for testing analytics,
machine learning, or reporting systems.

Features:
- Uses Faker to generate product names, IDs, and transaction references.
- Adds realism with seasonality, promotions, discounts, and biased date sampling.
- Outputs a Pandas DataFrame with 1,000+ rows, ready for analysis or export.
"""

import numpy as np
import pandas as pd
from faker import Faker

# --- Setup
SEED = 42
np.random.seed(SEED)
fake = Faker()

# Try to use Faker's commerce provider (if available) for product names
try:
    import importlib

    commerce_provider = importlib.import_module("faker.providers.commerce")
    fake.add_provider(commerce_provider)

    # Prefer product_name; fall back to ecommerce_name if available
    if hasattr(fake, "product_name"):
        product_name_fn = fake.product_name
    elif hasattr(fake, "ecommerce_name"):
        product_name_fn = fake.ecommerce_name
    else:
        raise AttributeError("No product name generator in commerce provider")
except (ModuleNotFoundError, AttributeError):
    # Fallback: use two random words capitalized if commerce provider/method is unavailable
    product_name_fn = lambda: f"{fake.word().capitalize()} {fake.word().capitalize()}"

# --- Config
N_ROWS = 1000  # number of synthetic rows to generate
start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)  # last 12 months

# Product categories with typical Mauritian retail price ranges (MUR)
categories = {
    "Electronics": (1500, 25000),
    "Home & Kitchen": (300, 8000),
    "Groceries": (20, 800),
    "Clothing": (200, 6000),
    "Beauty": (100, 4000),
    "Sports": (250, 9000),
    "Toys": (150, 3000),
    "Stationery": (20, 1200),
}

# Other categorical features
regions = ["North", "South", "East", "West", "Central"]
channels = ["In-Store", "Online"]
payment_methods = ["Card", "Cash", "Mobile Wallet"]


# --- Helper functions
def random_date():
    """
    Generates a random datetime with a bias towards recent days within the past year.

    This function computes a date within the past year by applying a squared random
    bias to favor more recent days. It generates a random hour and minute to
    produce a complete timestamp within the range of working hours (8:00 to 21:00).

    :return: A pandas Timestamp representing the generated random date and time.
    :rtype: pd.Timestamp
    """
    u = np.random.rand()
    biased = u ** 2  # more weight to recent days
    offset_days = int((1 - biased) * 365)
    d = start_date + pd.Timedelta(days=offset_days)
    hour = np.random.randint(8, 21)
    minute = np.random.randint(0, 60)
    return pd.Timestamp(d.date()) + pd.Timedelta(hours=hour, minutes=minute)


def draw_age():
    """
    Generates a randomly drawn age within a specific range derived from a normal
    distribution. The function uses a Gaussian distribution centered at 34 years
    with a standard deviation of 10 years and clips the result to a valid range of
    ages between 16 and 75, inclusive.

    :return: Randomly generated age constrained by the specified range
    :rtype: int
    """
    return int(np.clip(
        np.random.normal(34, 10),
        16,
        75)
    )


def draw_category():
    """
    Randomly selects and returns a category from the given predefined categories.

    The function uses a predefined probability distribution to select a category
    from the available options. Each category has a specific probability
    associated with it, represented by the `p` parameter passed to `np.random.choice`.

    :return: A randomly chosen category from the `categories` dictionary.
    :rtype: Any
    """
    return np.random.choice(
        list(categories.keys()),
        p=[0.16, 0.14, 0.18, 0.16, 0.10, 0.10, 0.08, 0.08]
    )


def draw_unit_price(cat):
    """
    Draws a unit price randomly from a given category using a log-normal distribution
    defined by a specified range for the category.

    This function calculates the natural logarithm of the midpoint of the provided range
    and uses it as the mean (mu) for a log-normal distribution. A fixed standard deviation
    (sigma) of 0.5 is used. The resulting value is clipped to fall within the specified
    low and high bounds for the category.

    :param cat: The category for which to draw the unit price. The category must
                exist in the `categories` dictionary that maps category names
                to tuples of (low, high) price bounds.
    :type cat: str
    :return: A randomly generated unit price clipped within the specified range for
             the provided category.
    :rtype: float
    """
    low, high = categories[cat]
    mu = np.log((low + high) / 2)
    sigma = 0.5
    price = np.exp(np.random.normal(mu, sigma))
    return float(np.clip(price, low, high))


def draw_quantity(cat):
    """
    Generate a randomized quantity value based on the specified category.

    This function computes a quantity for the given category. For specified categories
    (Groceries or Stationery), it uses a geometric distribution to determine the quantity,
    with a minimum value of 1 and a maximum value of 10. For all other categories, it
    uses a Poisson distribution with a minimum value of 1 and a maximum value of 5.

    :param cat: The category to determine the quantity for. Expected to be either
                "Groceries", "Stationery", or any other category string.
    :type cat: str
    :return: The computed quantity value based on the category.
    :rtype: int
    """
    if cat in ("Groceries", "Stationery"):
        return int(np.clip(
            np.random.geometric(p=0.45),
            1,
            10)
        )

    return int(np.clip(
        np.random.poisson(lam=1.4) + 1,
        1,
        5)
    )


def draw_discount_pct(dt, channel):
    """
    Calculate the discount percentage based on multiple conditions such as time of the
    month, day of the week, month, and sales channel. This function leverages randomness
    to simulate variable discounts under different conditions.

    :param dt: Datetime object representing the current date and time.
               It is used to evaluate day-specific, month-specific, and other time-dependent conditions.
    :type dt: datetime.datetime
    :param channel: Sales channel which can influence the discount value,
                    such as "Online" or other specific channel types.
    :type channel: str
    :return: A float value representing the discount percentage calculated based on
             given conditions, clipped to a maximum of 40%.
    :rtype: float
    """
    base = np.random.choice(
        [0.0, 0.05, 0.10, 0.15, 0.20],
        p=[0.70, 0.13, 0.10, 0.05, 0.02]
    )

    if dt.weekday() >= 5:  # weekend boost
        base += np.random.choice(
            [0.0, 0.05],
            p=[0.7, 0.3]
        )
    if dt.day >= 27:  # month-end sales
        base += np.random.choice(
            [0.0, 0.05],
            p=[0.7, 0.3]
        )
    if dt.month == 11:  # November sales (Black Friday equivalent)
        base += 0.05 if np.random.rand() < 0.4 else 0.0
    if channel == "Online":  # online vouchers
        base += np.random.choice(
            [0.0, 0.05],
            p=[0.6, 0.4]
        )
    return float(np.clip(
        base,
        0.0,
        0.40)
    )


def seasonality_multiplier(dt, cat):
    """
    Calculates a seasonal multiplier based on the product category and the month.

    The seasonal multiplier adjusts a base value depending on the month and the
    product category. Specific categories and months have predefined adjustment
    factors, enhancing the base value in peak seasons.

    :param dt: A datetime object representing the date to evaluate the month from.
    :type dt: datetime.datetime

    :param cat: The category of the product. Possible values include "Electronics",
        "Toys", "Stationery", and "Clothing".
    :type cat: str

    :return: A float representing the seasonal adjustment multiplier.
    :rtype: float
    """
    m, mult = dt.month, 1.0
    if cat in ("Electronics", "Toys") and m in (11, 12): mult *= 1.15
    if cat == "Stationery" and m in (1, 2): mult *= 1.20
    if cat == "Clothing" and m == 12: mult *= 1.10
    return mult


# --- Generate dataset
rows = []
for _ in range(N_ROWS):
    dt = random_date()  # Generate a random date
    category = draw_category()  # Draw a random category
    product = product_name_fn() if callable(product_name_fn) else fake.word().capitalize()  # Generate a random product name
    unit_price = draw_unit_price(category)  # Draw a random unit price for the category
    qty = draw_quantity(category)  # Draw a random quantity for the category
    channel = np.random.choice(channels, p=[0.7, 0.3])  # 70% in-store
    region = np.random.choice(regions, p=[0.22, 0.18, 0.18, 0.18, 0.24])
    pay = np.random.choice(payment_methods, p=[0.6, 0.25, 0.15])
    discount = draw_discount_pct(dt, channel)
    effective_price = unit_price * seasonality_multiplier(dt, category)
    sale_amount = float(np.round(
        effective_price * qty * (1.0 - discount),
        2
    ))

    # Append row
    rows.append({
        "Product Name": product,
        "Category": category,
        "Unit Price (MUR)": float(np.round(unit_price, 2)),
        "Quantity": qty,
        "Discount %": float(np.round(discount, 2)),
        "Sale Amount (MUR)": sale_amount,
        "Date of Sale": dt.normalize(),
        "Time of Sale": dt.time(),
        "Channel": channel,
        "Payment Method": pay,
        "Store Region": region,
        "Customer Age": draw_age(),
        "Customer ID": f"CUST-{fake.unique.random_int(min=10000, max=99999)}",
        "Transaction ID": f"TX-{fake.unique.random_int(min=10_000_000, max=99_999_999)}",
    })

# --- Create DataFrame
df = pd.DataFrame(rows)

# --- Save outputs
df.to_csv("retail_sales_synthetic.csv", index=False)

# Save parquet only if a compatible engine is installed
_parquet_ok = False
try:
    import pyarrow  # type: ignore  # noqa: F401

    _parquet_ok = True
except ModuleNotFoundError:
    try:
        import fastparquet  # type: ignore  # noqa: F401

        _parquet_ok = True
    except ModuleNotFoundError:
        _parquet_ok = False

if _parquet_ok:
    df.to_parquet("retail_sales_synthetic.parquet", index=False)

print(df.head())
print("\nRows generated:", len(df))
print("\nDtypes:\n", df.dtypes)
