from datetime import datetime, timedelta

import baostock as bs
import pandas as pd


def _result_to_dataframe(result) -> pd.DataFrame:
    rows = []

    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    return pd.DataFrame(rows, columns=result.fields)


def _market_board(baostock_code: str) -> str:
    if baostock_code.startswith("sh.688"):
        return "科创板"
    if baostock_code.startswith("sh.6"):
        return "沪市主板"
    if baostock_code.startswith("sz.300"):
        return "创业板"
    if baostock_code.startswith("sz.0"):
        return "深市主板"
    if baostock_code.startswith("bj."):
        return "北交所"
    return "其他"


def get_a_stock_universe(include_bj: bool = True, active_only: bool = True) -> pd.DataFrame:
    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_msg}")

    try:
        stock_df = pd.DataFrame()

        for day_offset in range(15):
            query_date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            stock_result = bs.query_all_stock(day=query_date)

            if stock_result.error_code != "0":
                continue

            stock_df = _result_to_dataframe(stock_result)

            if not stock_df.empty:
                break

        if stock_df.empty:
            raise RuntimeError("没有获取到 A 股股票池数据。")

        industry_result = bs.query_stock_industry()

        if industry_result.error_code == "0":
            industry_df = _result_to_dataframe(industry_result)
        else:
            industry_df = pd.DataFrame()

    finally:
        bs.logout()

    stock_df = stock_df.rename(columns={
        "code": "baostock代码",
        "code_name": "股票名称",
        "tradeStatus": "交易状态"
    })

    a_stock_prefixes = ("sh.6", "sz.0", "sz.3", "bj.")
    stock_df = stock_df[
        stock_df["baostock代码"].astype(str).str.startswith(a_stock_prefixes)
    ].copy()

    if not include_bj:
        stock_df = stock_df[
            ~stock_df["baostock代码"].astype(str).str.startswith("bj.")
        ].copy()

    if active_only and "交易状态" in stock_df.columns:
        stock_df = stock_df[stock_df["交易状态"].astype(str) == "1"].copy()

    stock_df["股票代码"] = stock_df["baostock代码"].map(lambda value: value.split(".")[-1])
    stock_df["市场板块"] = stock_df["baostock代码"].map(_market_board)

    if not industry_df.empty:
        industry_df = industry_df.rename(columns={
            "code": "baostock代码",
            "industry": "行业板块",
            "industryClassification": "行业分类"
        })

        industry_columns = [
            column
            for column in ["baostock代码", "行业板块", "行业分类"]
            if column in industry_df.columns
        ]

        stock_df = stock_df.merge(
            industry_df[industry_columns],
            on="baostock代码",
            how="left"
        )

    if "行业板块" not in stock_df.columns:
        stock_df["行业板块"] = "未知"

    if "行业分类" not in stock_df.columns:
        stock_df["行业分类"] = "未知"

    stock_df["股票名称"] = stock_df["股票名称"].fillna("未知")
    stock_df["行业板块"] = stock_df["行业板块"].fillna("未知")
    stock_df["行业分类"] = stock_df["行业分类"].fillna("未知")

    output_columns = [
        "股票代码",
        "股票名称",
        "市场板块",
        "行业板块",
        "行业分类",
        "baostock代码"
    ]

    return stock_df[output_columns].sort_values("股票代码").reset_index(drop=True)
