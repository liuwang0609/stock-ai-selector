import pandas as pd


def build_industry_activity_summary(
    stock_pool_df: pd.DataFrame,
    group_column: str = "行业板块"
) -> pd.DataFrame:
    required_columns = {group_column, "当日成交量", "当日成交额"}

    if stock_pool_df.empty or not required_columns.issubset(stock_pool_df.columns):
        return pd.DataFrame()

    source_df = stock_pool_df.copy()

    if "快照涨跌幅" not in source_df.columns:
        source_df["快照涨跌幅"] = pd.NA

    source_df["是否上涨"] = pd.to_numeric(
        source_df["快照涨跌幅"],
        errors="coerce"
    ) > 0

    summary_df = (
        source_df
        .groupby(group_column, dropna=False)
        .agg(
            行业股票数量=("股票代码", "count"),
            行业成交量=("当日成交量", "sum"),
            行业成交额=("当日成交额", "sum"),
            行业平均成交额=("当日成交额", "mean"),
            行业中位成交额=("当日成交额", "median"),
            行业平均涨跌幅=("快照涨跌幅", "mean"),
            行业内上涨占比=("是否上涨", "mean")
        )
        .reset_index()
    )

    summary_df["成交额分位"] = summary_df["行业成交额"].rank(
        pct=True,
        ascending=True
    ) * 100
    summary_df["平均成交额分位"] = summary_df["行业平均成交额"].rank(
        pct=True,
        ascending=True
    ) * 100
    summary_df["中位成交额分位"] = summary_df["行业中位成交额"].rank(
        pct=True,
        ascending=True
    ) * 100
    summary_df["涨跌幅分位"] = summary_df["行业平均涨跌幅"].rank(
        pct=True,
        ascending=True
    ) * 100
    summary_df["上涨占比分位"] = summary_df["行业内上涨占比"].rank(
        pct=True,
        ascending=True
    ) * 100

    summary_df["行业活跃度评分"] = (
        # 只保留中位成交额参与主评分，降低行业体量偏差
        summary_df["中位成交额分位"].fillna(50) * 0.25
        + summary_df["涨跌幅分位"].fillna(50) * 0.45
        + summary_df["上涨占比分位"].fillna(50) * 0.30
    ).round(2)

    return summary_df.sort_values(
        by="行业活跃度评分",
        ascending=False
    ).reset_index(drop=True)


def filter_top_industries(
    stock_pool_df: pd.DataFrame,
    top_industry_count: int,
    group_column: str = "行业板块"
) -> pd.DataFrame:
    summary_df = build_industry_activity_summary(
        stock_pool_df,
        group_column=group_column
    )

    if summary_df.empty:
        return stock_pool_df

    selected_industries = summary_df.head(top_industry_count)[group_column].tolist()

    return stock_pool_df[
        stock_pool_df[group_column].isin(selected_industries)
    ].copy()


def _rank_percentile_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series([pd.NA] * len(values), index=values.index)

    return values.rank(pct=True, ascending=True) * 100


def _rank_position_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series([pd.NA] * len(values), index=values.index)

    return values.rank(method="min", ascending=False)


def add_industry_competitiveness(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty or "行业板块" not in result_df.columns:
        return result_df

    result_df = result_df.copy()

    grouped = result_df.groupby("行业板块", dropna=False)

    result_df["行业内综合排名"] = grouped["综合评分"].transform(
        lambda group: group.rank(method="min", ascending=False)
    )
    if "主营收入" in result_df.columns:
        result_df["行业内营收排名"] = grouped["主营收入"].transform(_rank_position_values)
    else:
        result_df["行业内营收排名"] = pd.NA

    if "净利润" in result_df.columns:
        result_df["行业内净利润排名"] = grouped["净利润"].transform(_rank_position_values)
    else:
        result_df["行业内净利润排名"] = pd.NA

    percentile_columns = {
        "ROE行业分位": "ROE",
        "毛利率行业分位": "毛利率",
        "净利率行业分位": "净利率",
        "净利润增速行业分位": "净利润同比",
        "营收增速行业分位": "营收同比",
        "营收行业分位": "主营收入",
        "净利润行业分位": "净利润"
    }

    for output_column, source_column in percentile_columns.items():
        if source_column in result_df.columns:
            result_df[output_column] = grouped[source_column].transform(_rank_percentile_values)
        else:
            result_df[output_column] = pd.NA

    moat_columns = [
        column
        for column in [
            "毛利率行业分位",
            "ROE行业分位",
            "ROE稳定性评分",
            "毛利率稳定性评分"
        ]
        if column in result_df.columns
    ]

    if moat_columns:
        result_df["技术壁垒代理评分"] = result_df[moat_columns].mean(
            axis=1,
            skipna=True
        ).fillna(50).round(2)

    score_groups = [
        (["ROE行业分位", "毛利率行业分位", "净利率行业分位"], 0.35),
        (["净利润增速行业分位", "营收增速行业分位"], 0.30),
        (["营收行业分位", "净利润行业分位"], 0.20),
        (["技术壁垒代理评分"], 0.15),
    ]

    weighted_sum = pd.Series(0.0, index=result_df.index)
    weight_sum = pd.Series(0.0, index=result_df.index)

    for columns, weight in score_groups:
        available_columns = [
            column
            for column in columns
            if column in result_df.columns
        ]

        if not available_columns:
            continue

        component_score = result_df[available_columns].mean(axis=1, skipna=True)
        valid_mask = component_score.notna()

        weighted_sum = weighted_sum.add(
            component_score.fillna(0) * weight,
            fill_value=0
        )
        weight_sum = weight_sum.add(
            valid_mask.astype(float) * weight,
            fill_value=0
        )

    if weight_sum.gt(0).any():
        result_df["行业内竞争力评分"] = (
            weighted_sum / weight_sum.replace(0, pd.NA)
        ).fillna(50).round(2)
    else:
        result_df["行业内竞争力评分"] = 50.0

    def leadership_label(row):
        if row["行业内竞争力评分"] >= 80:
            return "强"
        if row["行业内竞争力评分"] >= 60:
            return "中"
        return "观察"

    result_df["细分龙头标签"] = result_df.apply(leadership_label, axis=1)

    return result_df
