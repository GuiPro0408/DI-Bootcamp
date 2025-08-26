"""
US Superstore — Marketing Strategy Analysis (Python 3.13)

What this script does
---------------------
- Loads the US Superstore dataset from an Excel file (.xls or .xlsx)
- Preprocesses columns (dates, numerics, trims strings)
- Answers business questions with tables & matplotlib charts (no seaborn/styles)
- Saves results (CSV + PNG) to ./outputs

Usage
-----
python superstore_marketing_analysis.py --file "US Superstore data.xls"

Notes
-----
- Charts use default matplotlib settings (no specific colors/styles as requested).
- "Outstanding customer in NY" is defined as the highest total Profit (assumption).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_any_excel(path: Path) -> pd.DataFrame:
    """
    Read .xls or .xlsx with pandas. For .xls, requires xlrd to be installed.
    Auto-detects the best sheet by required columns coverage (fallback: max rows).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        # Try loading all sheets
        sheets = pd.read_excel(path, sheet_name=None)  # engine auto
    except ImportError as e:
        # Common when reading .xls without xlrd installed
        tips = [
            "If the file is .xls, install xlrd: pip install xlrd",
            "Or convert the workbook to .xlsx and rerun this script."
        ]
        msg = f"Could not import engine to read Excel. {e}\n" + "\n".join(f"- {t}" for t in tips)
        raise RuntimeError(msg) from e
    except Exception as e:
        raise RuntimeError(f"Failed reading Excel file: {e}") from e

    required_cols = {
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
        "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region",
        "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"
    }

    def normalize_cols(cols):
        return [str(c).strip() for c in cols]

    def sheet_score(df):
        cols = set(normalize_cols(df.columns))
        return len(required_cols.intersection(cols)), len(df)

    best_sheet_name = None
    best_score = (-1, -1)

    for name, sdf in sheets.items():
        score = sheet_score(sdf)
        if score > best_score:
            best_score = score
            best_sheet_name = name

    df = sheets[best_sheet_name].copy()
    df.columns = normalize_cols(df.columns)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
        "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region",
        "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"
    ]
    present_cols = [c for c in required_cols if c in df.columns]
    df = df[present_cols].copy()

    # Dates
    for dcol in ["Order Date", "Ship Date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Numerics
    for ncol in ["Sales", "Quantity", "Discount", "Profit"]:
        if ncol in df.columns:
            df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    # Basic cleanup
    key_cols = ["Order ID", "Order Date", "Customer ID", "State", "City", "Sales", "Profit"]
    key_cols = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=key_cols).copy()

    # Strip
    for scol in ["State", "City", "Customer Name", "Segment", "Category", "Sub-Category", "Region", "Country"]:
        if scol in df.columns:
            df[scol] = df[scol].astype(str).str.strip()

    # Profitability metric
    df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    return df


def save_table(df: pd.DataFrame, outputs_dir: Path, name: str):
    out = outputs_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


def barplot(x, y, title, xlabel, ylabel, outputs_dir: Path, name: str, rotate=False):
    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    if rotate:
        plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = outputs_dir / f"{name}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved chart: {out}")


def lineplot(x, y, title, xlabel, ylabel, outputs_dir: Path, name: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = outputs_dir / f"{name}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved chart: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to US Superstore dataset (.xls or .xlsx)")
    ap.add_argument("--outputs", default="outputs", help="Output directory for CSV/PNG")
    args = ap.parse_args()

    data_path = Path(args.file)
    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = read_any_excel(data_path)
    df = preprocess(df)

    # Q1: States with the most sales
    state_sales = df.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    save_table(state_sales.head(15), outputs_dir, "top_states_by_sales")
    barplot(
        x=state_sales.head(15)["State"],
        y=state_sales.head(15)["Sales"],
        title="Top 15 States by Total Sales",
        xlabel="State", ylabel="Total Sales",
        outputs_dir=outputs_dir, name="top15_states_by_sales", rotate=True
    )

    # Q2: NY vs CA
    ny_ca = df[df["State"].isin(["New York", "California"])]
    ny_ca_summary = ny_ca.groupby("State", as_index=False)[["Sales", "Profit"]].sum()
    save_table(ny_ca_summary, outputs_dir, "ny_vs_ca_sales_profit")

    # Grouped bars in a single chart
    labels = ny_ca_summary["State"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, ny_ca_summary["Sales"], width, label="Sales")
    plt.bar(x + width / 2, ny_ca_summary["Profit"], width, label="Profit")
    plt.xticks(x, labels)
    plt.title("New York vs California — Total Sales & Profit")
    plt.xlabel("State")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    out = outputs_dir / "ny_vs_ca_sales_profit.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved chart: {out}")

    # Q3: Outstanding customer in New York (by Profit)
    ny = df[df["State"] == "New York"]
    ny_cust = ny.groupby("Customer Name", as_index=False).agg({"Sales": "sum", "Profit": "sum"}).sort_values("Profit",
                                                                                                             ascending=False)
    save_table(ny_cust.head(10), outputs_dir, "ny_top10_customers_by_profit")
    barplot(
        x=ny_cust.head(10)["Customer Name"],
        y=ny_cust.head(10)["Profit"],
        title="Top 10 New York Customers by Profit",
        xlabel="Customer Name", ylabel="Total Profit",
        outputs_dir=outputs_dir, name="ny_top10_customers_by_profit", rotate=True
    )

    outstanding_text = ""
    if not ny_cust.empty:
        first = ny_cust.iloc[0]
        outstanding_text = f"Outstanding NY customer (by Profit): {first['Customer Name']} — Profit={first['Profit']:,.2f}, Sales={first['Sales']:,.2f}"

    # Q4: Differences among states in profitability
    state_summary = df.groupby("State", as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    state_summary["Profit_Margin"] = np.where(state_summary["Sales"] != 0,
                                              state_summary["Profit"] / state_summary["Sales"], np.nan)
    save_table(state_summary, outputs_dir, "state_sales_profit_margin")

    threshold = state_summary["Sales"].median() * 0.25
    filtered = state_summary[state_summary["Sales"] >= threshold].copy()

    top_margin = filtered.sort_values("Profit_Margin", ascending=False).head(10)
    bottom_margin = filtered.sort_values("Profit_Margin", ascending=True).head(10)

    barplot(
        x=top_margin["State"], y=top_margin["Profit_Margin"],
        title="Top 10 States by Profit Margin (filtered)", xlabel="State", ylabel="Profit Margin",
        outputs_dir=outputs_dir, name="states_top10_profit_margin", rotate=True
    )
    barplot(
        x=bottom_margin["State"], y=bottom_margin["Profit_Margin"],
        title="Bottom 10 States by Profit Margin (filtered)", xlabel="State", ylabel="Profit Margin",
        outputs_dir=outputs_dir, name="states_bottom10_profit_margin", rotate=True
    )

    # Q5: Pareto on Customers & Profit
    cust_profit = df.groupby("Customer ID", as_index=False).agg({
        "Customer Name": "first",
        "Profit": "sum",
        "Sales": "sum"
    }).sort_values("Profit", ascending=False)

    total_profit = cust_profit["Profit"].sum()
    cust_profit["CumProfit"] = cust_profit["Profit"].cumsum()
    cust_profit["CumProfitShare"] = np.where(total_profit != 0, cust_profit["CumProfit"] / total_profit, np.nan)
    save_table(cust_profit, outputs_dir, "customers_profit_cumulative")

    if total_profit > 0:
        idx_80 = np.searchsorted(cust_profit["CumProfitShare"].values, 0.8, side="left")
        cust_needed_for_80 = idx_80 + 1
        share_top_20pct = cust_profit.head(max(1, int(0.2 * len(cust_profit))))["Profit"].sum() / total_profit
    else:
        cust_needed_for_80 = np.nan
        share_top_20pct = np.nan

    lineplot(
        x=np.arange(1, len(cust_profit) + 1), y=cust_profit["CumProfitShare"],
        title="Cumulative Profit Share by Customers (sorted by Profit)",
        xlabel="Number of Customers", ylabel="Cumulative Profit Share",
        outputs_dir=outputs_dir, name="cumulative_profit_share"
    )

    # Q6: Top 20 cities by Sales & Profit + differences
    city_summary = df.groupby(["City", "State"], as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    city_summary["Profit_Margin"] = np.where(city_summary["Sales"] != 0, city_summary["Profit"] / city_summary["Sales"],
                                             np.nan)

    top20_cities_sales = city_summary.sort_values("Sales", ascending=False).head(20)
    top20_cities_profit = city_summary.sort_values("Profit", ascending=False).head(20)
    save_table(top20_cities_sales, outputs_dir, "top20_cities_by_sales")
    save_table(top20_cities_profit, outputs_dir, "top20_cities_by_profit")

    barplot(
        x=top20_cities_sales["City"], y=top20_cities_sales["Sales"],
        title="Top 20 Cities by Sales", xlabel="City", ylabel="Total Sales",
        outputs_dir=outputs_dir, name="top20_cities_sales", rotate=True
    )
    barplot(
        x=top20_cities_profit["City"], y=top20_cities_profit["Profit"],
        title="Top 20 Cities by Profit", xlabel="City", ylabel="Total Profit",
        outputs_dir=outputs_dir, name="top20_cities_profit", rotate=True
    )

    # Q7: Top 20 customers by Sales
    cust_sales = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    top20_cust_sales = cust_sales.sort_values("Sales", ascending=False).head(20)
    save_table(top20_cust_sales, outputs_dir, "top20_customers_by_sales")
    barplot(
        x=top20_cust_sales["Customer Name"], y=top20_cust_sales["Sales"],
        title="Top 20 Customers by Sales", xlabel="Customer Name", ylabel="Total Sales",
        outputs_dir=outputs_dir, name="top20_customers_sales", rotate=True
    )

    # Q8: Cumulative Sales by Customers & Pareto
    cust_sales_sorted = cust_sales.sort_values("Sales", ascending=False).reset_index(drop=True).copy()
    total_sales = cust_sales_sorted["Sales"].sum()
    cust_sales_sorted["CumSales"] = cust_sales_sorted["Sales"].cumsum()
    cust_sales_sorted["CumSalesShare"] = np.where(total_sales != 0, cust_sales_sorted["CumSales"] / total_sales, np.nan)
    save_table(cust_sales_sorted, outputs_dir, "customers_sales_cumulative")

    if total_sales > 0:
        idx_sales_80 = np.searchsorted(cust_sales_sorted["CumSalesShare"].values, 0.8, side="left")
        cust_needed_for_80_sales = idx_sales_80 + 1
        share_top_20pct_sales = cust_sales_sorted.head(max(1, int(0.2 * len(cust_sales_sorted))))[
                                    "Sales"].sum() / total_sales
    else:
        cust_needed_for_80_sales = np.nan
        share_top_20pct_sales = np.nan

    lineplot(
        x=np.arange(1, len(cust_sales_sorted) + 1), y=cust_sales_sorted["CumSalesShare"],
        title="Cumulative Sales Share by Customers (sorted by Sales)",
        xlabel="Number of Customers", ylabel="Cumulative Sales Share",
        outputs_dir=outputs_dir, name="cumulative_sales_share"
    )

    # Q9: Marketing priorities (states & cities)
    def normalized(series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    states_scored = state_summary.copy()
    states_scored = states_scored.replace([np.inf, -np.inf], np.nan).dropna(subset=["Profit_Margin"])
    pos_margin = states_scored[states_scored["Profit_Margin"] > 0].copy()
    pos_margin["Profit_norm"] = normalized(pos_margin["Profit"])
    pos_margin["Margin_norm"] = normalized(pos_margin["Profit_Margin"])
    pos_margin["Score"] = pos_margin["Profit_norm"] * 0.7 + pos_margin["Margin_norm"] * 0.3
    priority_states = pos_margin.sort_values("Score", ascending=False).head(10)
    save_table(priority_states[["State", "Sales", "Profit", "Profit_Margin", "Score"]], outputs_dir,
               "priority_states_top10")

    cities_scored = city_summary.copy()
    cities_scored = cities_scored[cities_scored["Profit_Margin"] > 0].copy()
    cities_scored["Profit_norm"] = normalized(cities_scored["Profit"])
    cities_scored["Margin_norm"] = normalized(cities_scored["Profit_Margin"])
    cities_scored["Score"] = cities_scored["Profit_norm"] * 0.7 + cities_scored["Margin_norm"] * 0.3
    priority_cities = cities_scored.sort_values("Score", ascending=False).head(20)
    save_table(priority_cities[["City", "State", "Sales", "Profit", "Profit_Margin", "Score"]], outputs_dir,
               "priority_cities_top20")

    # Summary markdown
    md = []
    md.append("# US Superstore — Marketing Strategy Summary")
    md.append("")
    if "New York" in ny_ca_summary["State"].values and "California" in ny_ca_summary["State"].values:
        ny_sales = float(ny_ca_summary.loc[ny_ca_summary["State"] == "New York", "Sales"].values[0])
        ny_profit = float(ny_ca_summary.loc[ny_ca_summary["State"] == "New York", "Profit"].values[0])
        ca_sales = float(ny_ca_summary.loc[ny_ca_summary["State"] == "California", "Sales"].values[0])
        ca_profit = float(ny_ca_summary.loc[ny_ca_summary["State"] == "California", "Profit"].values[0])
        md.append(
            f"- **NY vs CA** — Sales: NY={ny_sales:,.2f}, CA={ca_sales:,.2f}; Profit: NY={ny_profit:,.2f}, CA={ca_profit:,.2f}")
    if outstanding_text:
        md.append(f"- {outstanding_text}")
    if total_profit > 0:
        md.append(
            f"- **Pareto (Profit)** — Top 20% customers contribute ~{share_top_20pct * 100:.1f}% of total profit; ~{(cust_needed_for_80 / len(cust_profit)) * 100:.1f}% of customers needed to reach 80%.")
    else:
        md.append("- **Pareto (Profit)** — Total profit <= 0 (edge case).")
    if total_sales > 0:
        md.append(
            f"- **Pareto (Sales)** — Top 20% customers contribute ~{share_top_20pct_sales * 100:.1f}% of total sales; ~{(cust_needed_for_80_sales / len(cust_sales_sorted)) * 100:.1f}% of customers needed to reach 80%.")
    else:
        md.append("- **Pareto (Sales)** — Total sales = 0 (edge case).")

    md.append("")
    md.append("## Priority Targets")
    md.append("### States (Top 10 by composite score)")
    md.append(priority_states[["State", "Sales", "Profit", "Profit_Margin", "Score"]].to_markdown(index=False))
    md.append("")
    md.append("### Cities (Top 20 by composite score)")
    md.append(priority_cities[["City", "State", "Sales", "Profit", "Profit_Margin", "Score"]].to_markdown(index=False))

    summary_path = outputs_dir / "marketing_strategy_summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\nDone. See CSV & PNG files under:", outputs_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
