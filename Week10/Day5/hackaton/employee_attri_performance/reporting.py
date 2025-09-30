from __future__ import annotations

from config import REPORTS_DIR


def generate_retention_recommendations(df, logger) -> None:
    """Create a simple text report with actionable insights."""
    logger.info("Compiling retention recommendations")
    attrition_rate = df["AttritionFlag"].mean() * 100
    overtime = df.groupby("OverTime")["AttritionFlag"].mean().mul(100)
    job_role = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
    worklife = df.groupby("WorkLifeBalance")["AttritionFlag"].mean().mul(100)
    job_satisfaction = df.groupby("JobSatisfaction")["AttritionFlag"].mean().mul(100)

    top_roles = job_role.head(3)
    low_balance_levels = worklife.sort_values(ascending=False).head(2)

    lines = [
        f"Overall attrition rate: {attrition_rate:.2f}%",
        "",
        "Groups with elevated attrition:",
    ]

    lines.extend([f"- {role}: {rate:.2f}%" for role, rate in top_roles.items()])

    if "Yes" in overtime.index and "No" in overtime.index:
        gap = overtime["Yes"] - overtime["No"]
        if abs(gap) > 5:
            lines.append(f"- Overtime status gap: {gap:.2f} percentage points (Yes minus No)")

    lines.append("")
    lines.append("Work-life balance levels with higher attrition:")
    lines.extend([f"- Level {int(level)}: {rate:.2f}%" for level, rate in low_balance_levels.items()])

    average_satisfaction = job_satisfaction.mean()
    high_attrition_satisfaction = job_satisfaction[job_satisfaction > average_satisfaction]
    if not high_attrition_satisfaction.empty:
        lines.append("")
        lines.append("Job satisfaction levels exceeding the average attrition rate:")
        lines.extend(
            [
                f"- Level {int(level)}: {rate:.2f}%"
                for level, rate in high_attrition_satisfaction.sort_values(ascending=False).items()
            ]
        )

    lines.append("")
    lines.append("Recommended focus areas:")
    lines.append("1. Review overtime policies for high-risk roles and monitor workload peaks.")
    lines.append("2. Design targeted development plans and mentoring for roles above the company average.")
    lines.append("3. Expand flexible work options where work-life balance ≤ 2 and track improvements.")
    lines.append(
        "4. Launch quarterly pulse surveys aligned with job satisfaction drivers to capture qualitative feedback."
    )

    recommendations_path = REPORTS_DIR / "retention_recommendations.txt"
    recommendations_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved retention recommendations to %s", recommendations_path)
