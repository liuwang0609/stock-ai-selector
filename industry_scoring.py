import pandas as pd


def build_industry_activity_summary(stock_pool_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"行业板块", "当日成交量", "当日成交额"}

    if stock_pool_df.empty or not required_columns.issubset(stock_pool_df.columns):
        return pd.DataFrame()

    summary_df = (
        stock_pool_df
        .groupby("行业板块", dropna=False)
        .agg(
            行业股票数量=("股票代码", "count"),
            行业成交量=("当日成交量", "sum"),
            行业成交额=("当日成交额", "sum"),
            行业平均成交额=("当日成交额", "mean")
        )
        .reset_index()
    )

    summary_df["行业活跃度评分"] = summary_df["行业成交额"].rank(
        pct=True,
        ascending=True
    ) * 100

    return summary_df.sort_values(
        by="行业活跃度评分",
        ascending=False
    ).reset_index(drop=True)


def filter_top_industries(stock_pool_df: pd.DataFrame, top_industry_count: int) -> pd.DataFrame:
    summary_df = build_industry_activity_summary(stock_pool_df)

    if summary_df.empty:
        return stock_pool_df

    selected_industries = summary_df.head(top_industry_count)["行业板块"].tolist()

    return stock_pool_df[
        stock_pool_df["行业板块"].isin(selected_industries)
    ].copy()


def _rank_percentile(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series([pd.NA] * len(group), index=group.index)

    values = pd.to_numeric(group[column], errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series([pd.NA] * len(group), index=group.index)

    return values.rank(pct=True, ascending=True) * 100


def _rank_position(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series([pd.NA] * len(group), index=group.index)

    values = pd.to_numeric(group[column], errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series([pd.NA] * len(group), index=group.index)

    return values.rank(method="min", ascending=False)


def add_industry_competitiveness(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty or "行业板块" not in result_df.columns:
        return result_df

    result_df = result_df.copy()

    grouped = result_df.groupby("行业板块", dropna=False)

    result_df["行业内综合排名"] = grouped["综合评分"].transform(
        lambda group: group.rank(method="min", ascending=False)
    )
    result_df["行业内营收排名"] = grouped.apply(
        lambda group: _rank_position(group, "主营收入"),
        include_groups=False
    ).reset_index(level=0, drop=True)
    result_df["行业内净利润排名"] = grouped.apply(
        lambda group: _rank_position(group, "净利润"),
        include_groups=False
    ).reset_index(level=0, drop=True)

    percentile_columns = {
        "ROE行业分位": "ROE",
        "毛利率行业分位": "毛利率",
        "净利率行业分位": "净利率",
        "净利润增速行业分位": "净利润同比",
        "营收行业分位": "主营收入",
        "净利润行业分位": "净利润"
    }

    for output_column, source_column in percentile_columns.items():
        result_df[output_column] = grouped.apply(
            lambda group, column=source_column: _rank_percentile(group, column),
            include_groups=False
        ).reset_index(level=0, drop=True)

    score_columns = [
        "ROE行业分位",
        "毛利率行业分位",
        "净利率行业分位",
        "净利润增速行业分位",
        "营收行业分位",
        "净利润行业分位"
    ]

    result_df["行业内竞争力评分"] = result_df[score_columns].mean(
        axis=1,
        skipna=True
    ).fillna(50).round(2)

    def leadership_label(row):
        if row["行业内竞争力评分"] >= 80:
            return "强"
        if row["行业内竞争力评分"] >= 60:
            return "中"
        return "观察"

    result_df["细分龙头标签"] = result_df.apply(leadership_label, axis=1)

    return result_df
