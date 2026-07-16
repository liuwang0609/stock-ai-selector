from datetime import datetime, timedelta

import akshare as ak
import baostock as bs
import pandas as pd


def to_baostock_code(symbol: str) -> str:
    symbol = symbol.strip().lower()

    if symbol.startswith(("sh.", "sz.", "bj.")):
        return symbol

    symbol = (
        symbol
        .replace(".sz", "")
        .replace(".sh", "")
        .replace(".bj", "")
        .replace("sz", "")
        .replace("sh", "")
        .replace("bj", "")
    )

    if symbol.startswith("6"):
        return f"sh.{symbol}"

    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"

    if symbol.startswith(("4", "8", "9")):
        return f"bj.{symbol}"

    return symbol


def to_akshare_daily_symbol(symbol: str) -> str:
    return to_baostock_code(symbol).replace(".", "")


def _standardize_history_dataframe(df: pd.DataFrame, code: str, days: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df["股票代码"] = code
    df = df.sort_values("日期")

    number_columns = ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "涨跌幅"]
    for column in number_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.tail(days)


def _get_stock_history_with_akshare(symbol: str, days: int) -> pd.DataFrame:
    ak_symbol = to_akshare_daily_symbol(symbol)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
    raw_df = ak.stock_zh_a_daily(
        symbol=ak_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if raw_df.empty:
        return raw_df

    df = raw_df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率"
    })

    if "换手率" in df.columns:
        df["换手率"] = pd.to_numeric(df["换手率"], errors="coerce") * 100

    df["涨跌幅"] = pd.to_numeric(df["收盘"], errors="coerce").pct_change() * 100

    return _standardize_history_dataframe(
        df=df,
        code=to_baostock_code(symbol),
        days=days
    )


def _get_stock_history_with_baostock(symbol: str, days: int) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
    code = to_baostock_code(symbol)

    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_msg}")

    try:
        result = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )

        if result.error_code != "0":
            raise RuntimeError(f"BaoStock 查询失败: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())

        df = pd.DataFrame(rows, columns=result.fields)

    finally:
        bs.logout()

    if df.empty:
        return df

    df = df.rename(columns={
        "date": "日期",
        "code": "股票代码",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
        "turn": "换手率",
        "pctChg": "涨跌幅"
    })

    return _standardize_history_dataframe(
        df=df,
        code=code,
        days=days
    )


def get_stock_history(symbol: str, days: int = 120) -> pd.DataFrame:
    try:
        return _get_stock_history_with_akshare(symbol=symbol, days=days)
    except Exception:
        return _get_stock_history_with_baostock(symbol=symbol, days=days)
