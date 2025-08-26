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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region",
    "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"
]
DATE_COLUMNS = ["Order Date", "Ship Date"]
NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]
STRING_COLUMNS = ["State", "City", "Customer Name", "Segment", "Category", "Sub-Category", "Region", "Country"]
KEY_COLUMNS = ["Order ID", "Order Date", "Customer ID", "State", "City", "Sales", "Profit"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_cols(cols: Iterable[Any]) -> list:
    return [str(c).strip() for c in cols]


def read_any_excel(path: Path) -> pd.DataFrame:
    """Read any Excel workbook and pick the *best* sheet (heuristic).

    Heuristic: maximize (#matching required columns, #rows).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        sheets = pd.read_excel(path, sheet_name=None)  # engine auto-selection
    except ImportError as e:  # Typically for legacy .xls lacking xlrd
        tips = [
            "If the file is .xls, install xlrd: pip install xlrd",
            "Or convert the workbook to .xlsx and rerun this script."
        ]
        msg = f"Could not import engine to read Excel. {e}\n" + "\n".join(f"- {t}" for t in tips)
        raise RuntimeError(msg) from e
    except Exception as e:  # pragma: no cover (broad safety)
        raise RuntimeError(f"Failed reading Excel file: {e}") from e

    if not sheets:
        raise RuntimeError("Workbook has no sheets")

    required_set = set(REQUIRED_COLUMNS)

    def sheet_score(df: pd.DataFrame) -> Tuple[int, int]:
        cols = set(_normalize_cols(df.columns))
        return len(required_set.intersection(cols)), len(df)

    best_name: Optional[str] = None
    best_score = (-1, -1)
    for name, sdf in sheets.items():
        score = sheet_score(sdf)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not determine a suitable sheet")

    df = sheets[best_name].copy()
    df.columns = _normalize_cols(df.columns)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns, coerce dtypes, create Profit_Margin, trim strings and drop incomplete rows."""
    present_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[present_cols].copy()

    for dcol in DATE_COLUMNS:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    for ncol in NUMERIC_COLUMNS:
        if ncol in df.columns:
            df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    drop_cols = [c for c in KEY_COLUMNS if c in df.columns]
    df = df.dropna(subset=drop_cols).copy()

    for scol in STRING_COLUMNS:
        if scol in df.columns:
            df[scol] = df[scol].astype(str).str.strip()

    if "Sales" in df.columns and "Profit" in df.columns:
        df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    return df


def save_table(df: pd.DataFrame, outputs_dir: Path, name: str) -> None:
    out = outputs_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


def barplot(x, y, title, xlabel, ylabel, outputs_dir: Path, name: str, rotate: bool = False) -> None:
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


def lineplot(x, y, title, xlabel, ylabel, outputs_dir: Path, name: str) -> None:
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


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _normalized(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.copy()
    mx, mn = series.max(), series.min()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def pareto_metrics(sorted_values: pd.Series) -> Dict[str, float]:
    """Given a descending-sorted metric series, compute Pareto style metrics.

    Returns keys: total, top20pct_share, customers_for_80pct, population
    Values are np.nan where not computable.
    """
    population = len(sorted_values)
    total = float(sorted_values.sum())
    if total <= 0 or population == 0:
        return {
            "total": total,
            "top20pct_share": np.nan,
            "customers_for_80pct": np.nan,
            "population": population,
        }
    cum = sorted_values.cumsum()
    cum_share = cum / total
    idx_80 = int(np.searchsorted(cum_share.values, 0.8, side="left"))  # 0-based
    customers_for_80pct = idx_80 + 1
    top_20_count = max(1, int(0.2 * population))
    top20pct_share = float(sorted_values.head(top_20_count).sum() / total)
    return {
        "total": total,
        "top20pct_share": top20pct_share,
        "customers_for_80pct": customers_for_80pct,
        "population": population,
    }


@dataclass
class AnalysisResults:
    ny_ca_summary: pd.DataFrame
    outstanding_text: str
    profit_pareto: Dict[str, float]
    sales_pareto: Dict[str, float]
    priority_states: pd.DataFrame
    priority_cities: pd.DataFrame
    cust_profit: pd.DataFrame
    cust_sales_sorted: pd.DataFrame


# ---------------------------------------------------------------------------
# Core analysis grouped for readability
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, outputs_dir: Path) -> AnalysisResults:
    # Q1 States by sales
    state_sales = df.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    save_table(state_sales.head(15), outputs_dir, "top_states_by_sales")
    barplot(state_sales.head(15)["State"], state_sales.head(15)["Sales"],
            "Top 15 States by Total Sales", "State", "Total Sales", outputs_dir, "top15_states_by_sales", rotate=True)

    # Q2 NY vs CA
    ny_ca = df[df["State"].isin(["New York", "California"])]
    ny_ca_summary = ny_ca.groupby("State", as_index=False)[["Sales", "Profit"]].sum()
    save_table(ny_ca_summary, outputs_dir, "ny_vs_ca_sales_profit")
    # Grouped bar chart
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

    # Q3 Outstanding customer in NY (by Profit)
    ny = df[df["State"] == "New York"]
    ny_cust = ny.groupby("Customer Name", as_index=False).agg({"Sales": "sum", "Profit": "sum"}) \
        .sort_values("Profit", ascending=False)
    save_table(ny_cust.head(10), outputs_dir, "ny_top10_customers_by_profit")
    barplot(ny_cust.head(10)["Customer Name"], ny_cust.head(10)["Profit"],
            "Top 10 New York Customers by Profit", "Customer Name", "Total Profit",
            outputs_dir, "ny_top10_customers_by_profit", rotate=True)
    outstanding_text = ""
    if not ny_cust.empty:
        first = ny_cust.iloc[0]
        outstanding_text = (f"Outstanding NY customer (by Profit): {first['Customer Name']} — "
                            f"Profit={first['Profit']:,.2f}, Sales={first['Sales']:,.2f}")

    # Q4 State profitability
    state_summary = df.groupby("State", as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    state_summary["Profit_Margin"] = np.where(state_summary["Sales"] != 0,
                                              state_summary["Profit"] / state_summary["Sales"], np.nan)
    save_table(state_summary, outputs_dir, "state_sales_profit_margin")

    threshold = state_summary["Sales"].median() * 0.25
    filtered = state_summary[state_summary["Sales"] >= threshold].copy()
    top_margin = filtered.sort_values("Profit_Margin", ascending=False).head(10)
    bottom_margin = filtered.sort_values("Profit_Margin", ascending=True).head(10)
    barplot(top_margin["State"], top_margin["Profit_Margin"],
            "Top 10 States by Profit Margin (filtered)", "State", "Profit Margin", outputs_dir,
            "states_top10_profit_margin", rotate=True)
    barplot(bottom_margin["State"], bottom_margin["Profit_Margin"],
            "Bottom 10 States by Profit Margin (filtered)", "State", "Profit Margin", outputs_dir,
            "states_bottom10_profit_margin", rotate=True)

    # Q5 Pareto Profit
    cust_profit = df.groupby("Customer ID", as_index=False).agg({
        "Customer Name": "first",
        "Profit": "sum",
        "Sales": "sum"
    }).sort_values("Profit", ascending=False)
    save_table(cust_profit, outputs_dir, "customers_profit_cumulative")
    cust_profit["CumProfit"] = cust_profit["Profit"].cumsum()
    total_profit = cust_profit["Profit"].sum()
    cust_profit["CumProfitShare"] = np.where(total_profit != 0, cust_profit["CumProfit"] / total_profit, np.nan)
    lineplot(np.arange(1, len(cust_profit) + 1), cust_profit["CumProfitShare"],
             "Cumulative Profit Share by Customers (sorted by Profit)",
             "Number of Customers", "Cumulative Profit Share", outputs_dir, "cumulative_profit_share")
    profit_pareto = pareto_metrics(cust_profit["Profit"])  # descending ensured

    # Q6 Top 20 cities by Sales & Profit
    city_summary = df.groupby(["City", "State"], as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    city_summary["Profit_Margin"] = np.where(city_summary["Sales"] != 0,
                                             city_summary["Profit"] / city_summary["Sales"], np.nan)
    top20_cities_sales = city_summary.sort_values("Sales", ascending=False).head(20)
    top20_cities_profit = city_summary.sort_values("Profit", ascending=False).head(20)
    save_table(top20_cities_sales, outputs_dir, "top20_cities_by_sales")
    save_table(top20_cities_profit, outputs_dir, "top20_cities_by_profit")
    barplot(top20_cities_sales["City"], top20_cities_sales["Sales"],
            "Top 20 Cities by Sales", "City", "Total Sales", outputs_dir, "top20_cities_sales", rotate=True)
    barplot(top20_cities_profit["City"], top20_cities_profit["Profit"],
            "Top 20 Cities by Profit", "City", "Total Profit", outputs_dir, "top20_cities_profit", rotate=True)

    # Q7 Top 20 customers by Sales
    cust_sales = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg({"Sales": "sum", "Profit": "sum"})
    top20_cust_sales = cust_sales.sort_values("Sales", ascending=False).head(20)
    save_table(top20_cust_sales, outputs_dir, "top20_customers_by_sales")
    barplot(top20_cust_sales["Customer Name"], top20_cust_sales["Sales"],
            "Top 20 Customers by Sales", "Customer Name", "Total Sales", outputs_dir, "top20_customers_sales", rotate=True)

    # Q8 Pareto Sales
    cust_sales_sorted = cust_sales.sort_values("Sales", ascending=False).reset_index(drop=True).copy()
    save_table(cust_sales_sorted, outputs_dir, "customers_sales_cumulative")
    cust_sales_sorted["CumSales"] = cust_sales_sorted["Sales"].cumsum()
    total_sales = cust_sales_sorted["Sales"].sum()
    cust_sales_sorted["CumSalesShare"] = np.where(total_sales != 0, cust_sales_sorted["CumSales"] / total_sales, np.nan)
    lineplot(np.arange(1, len(cust_sales_sorted) + 1), cust_sales_sorted["CumSalesShare"],
             "Cumulative Sales Share by Customers (sorted by Sales)",
             "Number of Customers", "Cumulative Sales Share", outputs_dir, "cumulative_sales_share")
    sales_pareto = pareto_metrics(cust_sales_sorted["Sales"])  # descending ensured

    # Q9 Marketing priorities (states & cities)
    states_scored = state_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["Profit_Margin"])  # copy not needed
    pos_margin = states_scored[states_scored["Profit_Margin"] > 0].copy()
    pos_margin["Profit_norm"] = _normalized(pos_margin["Profit"])
    pos_margin["Margin_norm"] = _normalized(pos_margin["Profit_Margin"])
    pos_margin["Score"] = pos_margin["Profit_norm"] * 0.7 + pos_margin["Margin_norm"] * 0.3
    priority_states = pos_margin.sort_values("Score", ascending=False).head(10)
    save_table(priority_states[["State", "Sales", "Profit", "Profit_Margin", "Score"]], outputs_dir,
               "priority_states_top10")

    cities_scored = city_summary[city_summary["Profit_Margin"] > 0].copy()
    cities_scored["Profit_norm"] = _normalized(cities_scored["Profit"])
    cities_scored["Margin_norm"] = _normalized(cities_scored["Profit_Margin"])
    cities_scored["Score"] = cities_scored["Profit_norm"] * 0.7 + cities_scored["Margin_norm"] * 0.3
    priority_cities = cities_scored.sort_values("Score", ascending=False).head(20)
    save_table(priority_cities[["City", "State", "Sales", "Profit", "Profit_Margin", "Score"]], outputs_dir,
               "priority_cities_top20")

    return AnalysisResults(
        ny_ca_summary=ny_ca_summary,
        outstanding_text=outstanding_text,
        profit_pareto=profit_pareto,
        sales_pareto=sales_pareto,
        priority_states=priority_states,
        priority_cities=priority_cities,
        cust_profit=cust_profit,
        cust_sales_sorted=cust_sales_sorted,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary(results: AnalysisResults, outputs_dir: Path) -> Path:
    md: list[str] = ["# US Superstore — Marketing Strategy Summary", ""]

    ny_ca_summary = results.ny_ca_summary
    if set(["New York", "California"]).issubset(set(ny_ca_summary["State"].values)):
        ny_sales = float(ny_ca_summary.loc[ny_ca_summary["State"] == "New York", "Sales"].values[0])
        ny_profit = float(ny_ca_summary.loc[ny_ca_summary["State"] == "New York", "Profit"].values[0])
        ca_sales = float(ny_ca_summary.loc[ny_ca_summary["State"] == "California", "Sales"].values[0])
        ca_profit = float(ny_ca_summary.loc[ny_ca_summary["State"] == "California", "Profit"].values[0])
        md.append(f"- **NY vs CA** — Sales: NY={ny_sales:,.2f}, CA={ca_sales:,.2f}; Profit: NY={ny_profit:,.2f}, CA={ca_profit:,.2f}")

    if results.outstanding_text:
        md.append(f"- {results.outstanding_text}")

    # Profit Pareto
    p = results.profit_pareto
    if p["total"] > 0:
        md.append(
            f"- **Pareto (Profit)** — Top 20% customers contribute ~{p['top20pct_share'] * 100:.1f}% of total profit; ~{(p['customers_for_80pct'] / p['population']) * 100:.1f}% of customers needed to reach 80%."
        )
    else:
        md.append("- **Pareto (Profit)** — Total profit <= 0 (edge case).")

    # Sales Pareto
    s = results.sales_pareto
    if s["total"] > 0:
        md.append(
            f"- **Pareto (Sales)** — Top 20% customers contribute ~{s['top20pct_share'] * 100:.1f}% of total sales; ~{(s['customers_for_80pct'] / s['population']) * 100:.1f}% of customers needed to reach 80%."
        )
    else:
        md.append("- **Pareto (Sales)** — Total sales = 0 (edge case).")

    md.append("")
    md.append("## Priority Targets")
    md.append("### States (Top 10 by composite score)")
    md.append(results.priority_states[["State", "Sales", "Profit", "Profit_Margin", "Score"]].to_markdown(index=False))
    md.append("")
    md.append("### Cities (Top 20 by composite score)")
    md.append(results.priority_cities[["City", "State", "Sales", "Profit", "Profit_Margin", "Score"]].to_markdown(index=False))

    summary_path = outputs_dir / "marketing_strategy_summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="US Superstore marketing strategy analysis")
    ap.add_argument("--file", required=True, help="Path to US Superstore dataset (.xls or .xlsx)")
    ap.add_argument("--outputs", default="outputs", help="Output directory for CSV/PNG")
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    data_path = Path(args.file)
    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = read_any_excel(data_path)
    df = preprocess(df)

    results = run_analysis(df, outputs_dir)
    build_summary(results, outputs_dir)

    print("\nDone. See CSV & PNG files under:", outputs_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover (CLI safety)
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
