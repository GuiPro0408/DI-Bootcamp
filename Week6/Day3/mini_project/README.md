# US Superstore — Marketing Strategy Analysis

This package contains a ready-to-run Python 3.13 script to analyze the **US Superstore** dataset and produce tables and charts to guide marketing decisions.

## Files
- `superstore_marketing_analysis.py` — main script
- `requirements.txt` — dependencies
- Outputs saved to `./outputs` (CSV + PNG + summary markdown)

## How to Run (locally)
1. **Install Python 3.13+** (3.11+ is usually fine).
2. Create a virtual environment *(recommended)* and activate it.
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script, pointing to your Excel file (.xls or .xlsx):
   ```bash
   python superstore_marketing_analysis.py --file "US Superstore data.xls"
   ```
   > If your workbook is `.xls`, ensure `xlrd` is installed (included in `requirements.txt`). If you still face issues, convert the file to `.xlsx` and rerun.

## What You Get
- Top states by sales (CSV + bar chart)
- NY vs CA sales/profit (table + grouped bar chart)
- Outstanding NY customer (by profit) and Top 10 list (CSV + bar chart)
- States profitability analysis (CSV + top/bottom margin charts)
- Pareto analysis of customers vs profit & sales (cumulative CSV + line charts)
- Top 20 cities by sales and profit (CSV + bar charts)
- Top 20 customers by sales (CSV + bar chart)
- Priority targets (states & cities) with a composite score (CSV)
- `marketing_strategy_summary.md` — compact summary with key findings

## Notes
- Charts use default matplotlib settings (no seaborn; no custom styles/colors).
- "Outstanding customer in NY" = highest total **Profit** (assumption).
- You can tweak the sales threshold used to filter states for margin comparison inside the script if needed.
