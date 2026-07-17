from datetime import datetime, timedelta

import akshare as ak
import baostock as bs
import pandas as pd

from data_loader import to_baostock_code


def _result_to_dataframe(result) -> pd.DataFrame:
    rows = []

    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    return pd.DataFrame(rows, columns=result.fields)


def _normalize_spot_code(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("sh", "").replace("sz", "").replace("bj", "")
    value = value.replace(".", "")
    return value[-6:]


def _rank_with_akshare_spot(stock_pool_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    spot_df = ak.stock_zh_a_spot()

    if spot_df.empty:
        raise RuntimeError("AKShare 全市场快照为空。")

    spot_df = spot_df.rename(columns={
        "代码": "原始代码",
        "名称": "快照名称",
        "最新价": "快照最新价",
        "涨跌幅": "快照涨跌幅",
        "成交量": "当日成交量",
        "成交额": "当日成交额",
        "时间戳": "快照时间"
    })

    spot_df["股票代码"] = spot_df["原始代码"].map(_normalize_spot_code)
    spot_df["快照涨跌幅"] = pd.to_numeric(spot_df["快照涨跌幅"], errors="coerce")
    spot_df["当日成交量"] = pd.to_numeric(spot_df["当日成交量"], errors="coerce")
    spot_df["当日成交额"] = pd.to_numeric(spot_df["当日成交额"], errors="coerce")

    merged_df = stock_pool_df.merge(
        spot_df[["股票代码", "快照最新价", "快照涨跌幅", "当日成交量", "当日成交额", "快照时间"]],
        on="股票代码",
        how="inner"
    )

    if merged_df.empty:
        raise RuntimeError("全市场快照与当前股票池没有匹配结果。")

    merged_df["最近交易日"] = datetime.now().strftime("%Y-%m-%d")

    merged_df = merged_df.sort_values(
        by="当日成交量",
        ascending=False
    )

    return merged_df.head(top_n).reset_index(drop=True)


def _rank_with_baostock_history(
    stock_pool_df: pd.DataFrame,
    top_n: int,
    lookback_days: int,
    progress_callback=None
) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_msg}")

    records = []
    total = len(stock_pool_df)

    try:
        for index, (_, stock_row) in enumerate(stock_pool_df.iterrows(), start=1):
            stock_code = str(stock_row["股票代码"])
            baostock_code = to_baostock_code(stock_code)

            result = bs.query_history_k_data_plus(
                baostock_code,
                "date,code,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )

            if result.error_code == "0":
                history_df = _result_to_dataframe(result)

                if not history_df.empty:
                    latest = history_df.tail(1).iloc[0]

                    record = stock_row.to_dict()
                    record.update({
                        "最近交易日": latest["date"],
                        "快照最新价": pd.to_numeric(latest["close"], errors="coerce"),
                        "快照涨跌幅": pd.NA,
                        "当日成交量": pd.to_numeric(latest["volume"], errors="coerce"),
                        "当日成交额": pd.to_numeric(latest["amount"], errors="coerce"),
                        "快照时间": "BaoStock日线"
                    })
                    records.append(record)

            if progress_callback:
                progress_callback(index, total)

    finally:
        bs.logout()

    volume_df = pd.DataFrame(records)

    if volume_df.empty:
        return volume_df

    volume_df = volume_df.sort_values(
        by="当日成交量",
        ascending=False
    )

    return volume_df.head(top_n).reset_index(drop=True)


def rank_stocks_by_latest_volume(
    stock_pool_df: pd.DataFrame,
    top_n: int = 200,
    lookback_days: int = 15,
    progress_callback=None
) -> pd.DataFrame:
    try:
        ranked_df = _rank_with_akshare_spot(
            stock_pool_df=stock_pool_df,
            top_n=top_n
        )
        ranked_df["成交量来源"] = "AKShare全市场快照"
        return ranked_df

    except Exception:
        ranked_df = _rank_with_baostock_history(
            stock_pool_df=stock_pool_df,
            top_n=top_n,
            lookback_days=lookback_days,
            progress_callback=progress_callback
        )

        if not ranked_df.empty:
            ranked_df["成交量来源"] = "BaoStock日线回退"

        return ranked_df
