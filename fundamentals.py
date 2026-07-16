from datetime import datetime

import akshare as ak
import baostock as bs
import pandas as pd

from data_loader import to_baostock_code


def _result_to_dataframe(result) -> pd.DataFrame:
    rows = []

    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    return pd.DataFrame(rows, columns=result.fields)


def _to_number(value):
    return pd.to_numeric(value, errors="coerce")


def _percent_to_ratio(value):
    value = _to_number(value)

    if pd.isna(value):
        return value

    return value / 100


def _plain_symbol(symbol: str) -> str:
    return to_baostock_code(symbol).split(".")[-1]


def _recent_periods(years_back: int = 3):
    current_year = datetime.now().year

    for year in range(current_year, current_year - years_back, -1):
        for quarter in [4, 3, 2, 1]:
            yield year, quarter


def _query_financial_dataframe(query_func, code: str, year: int, quarter: int) -> pd.DataFrame:
    result = query_func(code=code, year=year, quarter=quarter)

    if result.error_code != "0":
        return pd.DataFrame()

    return _result_to_dataframe(result)


def _extract_financial_abstract_value(df: pd.DataFrame, metric_name: str, period: str):
    metric_col = df.columns[1]
    matched_df = df[df[metric_col] == metric_name]

    if matched_df.empty or period not in matched_df.columns:
        return pd.NA

    return _to_number(matched_df.iloc[0][period])


def _get_fundamental_snapshot_with_akshare(symbol: str) -> dict:
    df = ak.stock_financial_abstract(symbol=_plain_symbol(symbol))

    if df.empty or len(df.columns) < 3:
        return {}

    period_columns = [
        column
        for column in df.columns[2:]
        if str(column).isdigit()
    ]

    if not period_columns:
        return {}

    latest_period = str(period_columns[0])

    return {
        "财报年份": int(latest_period[:4]),
        "财报季度": int((int(latest_period[4:6]) + 2) / 3),
        "公告日期": None,
        "财报日期": f"{latest_period[:4]}-{latest_period[4:6]}-{latest_period[6:8]}",
        "ROE": _percent_to_ratio(
            _extract_financial_abstract_value(df, "净资产收益率(ROE)", latest_period)
        ),
        "净利率": _percent_to_ratio(
            _extract_financial_abstract_value(df, "销售净利率", latest_period)
        ),
        "毛利率": _percent_to_ratio(
            _extract_financial_abstract_value(df, "毛利率", latest_period)
        ),
        "净利润": _extract_financial_abstract_value(df, "归母净利润", latest_period),
        "每股收益TTM": _extract_financial_abstract_value(df, "基本每股收益", latest_period),
        "主营收入": _extract_financial_abstract_value(df, "营业总收入", latest_period),
        "净利润同比": _percent_to_ratio(
            _extract_financial_abstract_value(df, "归属母公司净利润增长率", latest_period)
        ),
        "EPS同比": pd.NA,
        "净资产同比": pd.NA,
        "资产同比": pd.NA,
        "资产负债率": _percent_to_ratio(
            _extract_financial_abstract_value(df, "资产负债率", latest_period)
        ),
        "经营现金流净利润比": _extract_financial_abstract_value(
            df,
            "经营活动净现金/归属母公司的净利润",
            latest_period
        )
    }


def _get_fundamental_snapshot_with_baostock(symbol: str) -> dict:
    code = to_baostock_code(symbol)
    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_msg}")

    try:
        for year, quarter in _recent_periods():
            profit_df = _query_financial_dataframe(bs.query_profit_data, code, year, quarter)

            if profit_df.empty:
                continue

            growth_df = _query_financial_dataframe(bs.query_growth_data, code, year, quarter)
            balance_df = _query_financial_dataframe(bs.query_balance_data, code, year, quarter)
            cash_flow_df = _query_financial_dataframe(bs.query_cash_flow_data, code, year, quarter)

            profit = profit_df.iloc[0]
            growth = growth_df.iloc[0] if not growth_df.empty else {}
            balance = balance_df.iloc[0] if not balance_df.empty else {}
            cash_flow = cash_flow_df.iloc[0] if not cash_flow_df.empty else {}

            return {
                "财报年份": year,
                "财报季度": quarter,
                "公告日期": profit.get("pubDate"),
                "财报日期": profit.get("statDate"),
                "ROE": _to_number(profit.get("roeAvg")),
                "净利率": _to_number(profit.get("npMargin")),
                "毛利率": _to_number(profit.get("gpMargin")),
                "净利润": _to_number(profit.get("netProfit")),
                "每股收益TTM": _to_number(profit.get("epsTTM")),
                "主营收入": _to_number(profit.get("MBRevenue")),
                "净利润同比": _to_number(growth.get("YOYNI")),
                "EPS同比": _to_number(growth.get("YOYEPSBasic")),
                "净资产同比": _to_number(growth.get("YOYEquity")),
                "资产同比": _to_number(growth.get("YOYAsset")),
                "资产负债率": _to_number(balance.get("liabilityToAsset")),
                "经营现金流净利润比": _to_number(cash_flow.get("CFOToNP"))
            }

        return {}

    finally:
        bs.logout()


def get_fundamental_snapshot(symbol: str) -> dict:
    try:
        snapshot = _get_fundamental_snapshot_with_akshare(symbol)

        if snapshot:
            return snapshot

    except Exception:
        pass

    return _get_fundamental_snapshot_with_baostock(symbol)


def score_fundamentals(snapshot: dict) -> dict:
    if not snapshot:
        return {
            "score": 0,
            "grade": "无数据",
            "reasons": [],
            "risks": ["没有获取到可用的基本面数据。"]
        }

    score = 0
    reasons = []
    risks = []

    roe = snapshot.get("ROE")
    if pd.notna(roe):
        if roe >= 0.15:
            score += 25
            reasons.append("ROE 较高，盈利能力较强。")
        elif roe >= 0.08:
            score += 15
            reasons.append("ROE 处于可接受区间。")
        elif roe > 0:
            score += 5
            risks.append("ROE 偏低，盈利效率一般。")
        else:
            risks.append("ROE 为负，盈利能力存在压力。")

    yoy_ni = snapshot.get("净利润同比")
    if pd.notna(yoy_ni):
        if yoy_ni >= 0.2:
            score += 20
            reasons.append("净利润同比增长较快，成长性较好。")
        elif yoy_ni >= 0:
            score += 10
            reasons.append("净利润同比保持正增长。")
        else:
            risks.append("净利润同比下滑，成长性承压。")

    np_margin = snapshot.get("净利率")
    if pd.notna(np_margin):
        if np_margin >= 0.15:
            score += 15
            reasons.append("净利率较高，利润质量较好。")
        elif np_margin >= 0.05:
            score += 8
            reasons.append("净利率处于正向区间。")
        else:
            risks.append("净利率偏低，盈利质量一般。")

    cfo_to_np = snapshot.get("经营现金流净利润比")
    if pd.notna(cfo_to_np):
        if cfo_to_np >= 1:
            score += 15
            reasons.append("经营现金流对净利润覆盖较好。")
        elif cfo_to_np > 0:
            score += 8
            reasons.append("经营现金流为正。")
        else:
            risks.append("经营现金流表现偏弱。")

    liability_to_asset = snapshot.get("资产负债率")
    if pd.notna(liability_to_asset):
        if liability_to_asset <= 0.6:
            score += 15
            reasons.append("资产负债率处于相对健康区间。")
        elif liability_to_asset <= 0.75:
            score += 8
            risks.append("资产负债率略高，需要关注偿债压力。")
        else:
            risks.append("资产负债率较高，财务杠杆风险需要关注。")

    eps_ttm = snapshot.get("每股收益TTM")
    if pd.notna(eps_ttm):
        if eps_ttm > 0:
            score += 10
            reasons.append("每股收益 TTM 为正。")
        else:
            risks.append("每股收益 TTM 为负。")

    score = min(score, 100)

    if score >= 80:
        grade = "优秀"
    elif score >= 65:
        grade = "较好"
    elif score >= 50:
        grade = "一般"
    else:
        grade = "偏弱"

    return {
        "score": score,
        "grade": grade,
        "reasons": reasons,
        "risks": risks
    }
