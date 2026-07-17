import pandas as pd

from data_loader import get_stock_history
from fundamentals import get_fundamental_snapshot, score_fundamentals
from indicators import add_technical_indicators


DEFAULT_STOCK_POOL = [
    "000001", "000002", "000333", "000651", "000858",
    "002415", "002594", "300059", "300750", "600000",
    "600030", "600036", "600276", "600519", "600887",
    "601318", "601398", "601888", "603259", "688981"
]


def is_valid(value) -> bool:
    return not pd.isna(value)


def score_stock(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "score": 0,
            "grade": "无数据",
            "reasons": [],
            "risks": ["没有获取到有效数据。"]
        }

    clean_df = df.dropna(subset=[
        "收盘",
        "MA5",
        "MA10",
        "MA20",
        "MA60",
        "成交量",
        "成交量MA5",
        "成交量MA20",
        "RSI14",
        "MACD_DIF",
        "MACD_DEA",
        "MACD"
    ])

    if clean_df.empty:
        return {
            "score": 0,
            "grade": "数据不足",
            "reasons": [],
            "risks": ["历史数据太少，暂时无法计算完整技术指标。"]
        }

    latest = clean_df.iloc[-1]

    score = 0
    reasons = []
    risks = []

    close = latest["收盘"]

    if close > latest["MA20"]:
        score += 15
        reasons.append("收盘价站上 MA20，短中期趋势较强。")
    else:
        risks.append("收盘价低于 MA20，短中期趋势偏弱。")

    if close > latest["MA60"]:
        score += 15
        reasons.append("收盘价站上 MA60，中期趋势保持向上。")
    else:
        risks.append("收盘价低于 MA60，中期趋势仍需观察。")

    if latest["MA5"] > latest["MA10"]:
        score += 10
        reasons.append("MA5 高于 MA10，短线走势偏强。")
    else:
        risks.append("MA5 未高于 MA10，短线动能一般。")

    if latest["MA20"] > latest["MA60"]:
        score += 10
        reasons.append("MA20 高于 MA60，中期均线结构较好。")
    else:
        risks.append("MA20 未高于 MA60，中期均线结构尚未走强。")

    rsi = latest["RSI14"]

    if 45 <= rsi <= 70:
        score += 15
        reasons.append("RSI 处于相对健康区间，动量较好且未明显过热。")
    elif 70 < rsi <= 80:
        score += 8
        reasons.append("RSI 偏强，说明动量较高。")
        risks.append("RSI 已偏高，短线可能存在追高风险。")
    elif rsi > 80:
        risks.append("RSI 过高，短线过热风险较大。")
    else:
        risks.append("RSI 偏弱，动量不足。")

    if latest["MACD_DIF"] > latest["MACD_DEA"]:
        score += 10
        reasons.append("MACD DIF 高于 DEA，动能结构偏多。")
    else:
        risks.append("MACD DIF 未高于 DEA，动能结构一般。")

    if latest["MACD"] > 0:
        score += 5
        reasons.append("MACD 柱体为正，短期动能较好。")
    else:
        risks.append("MACD 柱体为负，短期动能偏弱。")

    if latest["成交量MA5"] > latest["成交量MA20"]:
        score += 10
        reasons.append("短期成交量高于中期成交量，资金活跃度提升。")
    else:
        risks.append("短期成交量未明显放大，资金活跃度一般。")

    recent_df = clean_df.tail(20)
    recent_high = recent_df["收盘"].max()

    if is_valid(recent_high) and recent_high > 0:
        drawdown = close / recent_high - 1

        if drawdown > -0.08:
            score += 10
            reasons.append("距离近 20 日高点回撤较小，走势相对稳健。")
        elif drawdown < -0.15:
            risks.append("距离近 20 日高点回撤较大，趋势修复仍需时间。")

    score = min(score, 100)

    if score >= 80:
        grade = "强势"
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


def analyze_technical_for_screening(symbol: str, days: int = 120) -> dict:
    df = get_stock_history(symbol=symbol, days=days)

    if df.empty:
        return {
            "股票代码": symbol,
            "综合评分": 0,
            "技术评分": 0,
            "基本面评分": 0,
            "技术等级": "无数据",
            "基本面等级": "待复评",
            "最新收盘价": None,
            "MA20": None,
            "RSI14": None,
            "MACD": None,
            "财报日期": None,
            "ROE": None,
            "净利润同比": None,
            "净利率": None,
            "毛利率": None,
            "净利润": None,
            "主营收入": None,
            "营收同比": None,
            "ROE均值": None,
            "ROE波动": None,
            "ROE稳定性评分": None,
            "毛利率均值": None,
            "毛利率波动": None,
            "毛利率稳定性评分": None,
            "经营现金流净利润比": None,
            "资产负债率": None,
            "_technical_reasons": [],
            "_technical_risks": ["没有获取到有效行情数据。"]
        }

    df = add_technical_indicators(df)
    technical_analysis = score_stock(df)
    latest = df.iloc[-1]

    return {
        "股票代码": symbol,
        "综合评分": technical_analysis["score"],
        "技术评分": technical_analysis["score"],
        "基本面评分": 0,
        "技术等级": technical_analysis["grade"],
        "基本面等级": "待复评",
        "最新收盘价": latest["收盘"],
        "MA20": latest["MA20"],
        "RSI14": latest["RSI14"],
        "MACD": latest["MACD"],
        "财报日期": None,
        "ROE": None,
        "净利润同比": None,
        "净利率": None,
        "毛利率": None,
        "净利润": None,
        "主营收入": None,
        "营收同比": None,
        "ROE均值": None,
        "ROE波动": None,
        "ROE稳定性评分": None,
        "毛利率均值": None,
        "毛利率波动": None,
        "毛利率稳定性评分": None,
        "经营现金流净利润比": None,
        "资产负债率": None,
        "_technical_reasons": technical_analysis["reasons"],
        "_technical_risks": technical_analysis["risks"]
    }


def attach_fundamental_review(
    record: dict,
    technical_weight: float = 0.6,
    fundamental_weight: float = 0.4
) -> dict:
    symbol = record["股票代码"]

    try:
        fundamental_snapshot = get_fundamental_snapshot(symbol)
        fundamental_analysis = score_fundamentals(fundamental_snapshot)
    except Exception as error:
        fundamental_snapshot = {}
        fundamental_analysis = {
            "score": 0,
            "grade": "错误",
            "reasons": [],
            "risks": [f"基本面数据获取失败: {error}"]
        }

    technical_score = record["技术评分"]
    fundamental_score = fundamental_analysis["score"]
    combined_score = round(
        technical_score * technical_weight + fundamental_score * fundamental_weight,
        2
    )

    reasons = record.get("_technical_reasons", []) + fundamental_analysis["reasons"]
    risks = record.get("_technical_risks", []) + fundamental_analysis["risks"]

    record.update({
        "综合评分": combined_score,
        "基本面评分": fundamental_score,
        "基本面等级": fundamental_analysis["grade"],
        "财报日期": fundamental_snapshot.get("财报日期"),
        "ROE": fundamental_snapshot.get("ROE"),
        "净利润同比": fundamental_snapshot.get("净利润同比"),
        "净利率": fundamental_snapshot.get("净利率"),
        "毛利率": fundamental_snapshot.get("毛利率"),
        "净利润": fundamental_snapshot.get("净利润"),
        "主营收入": fundamental_snapshot.get("主营收入"),
        "营收同比": fundamental_snapshot.get("营收同比"),
        "ROE均值": fundamental_snapshot.get("ROE均值"),
        "ROE波动": fundamental_snapshot.get("ROE波动"),
        "ROE稳定性评分": fundamental_snapshot.get("ROE稳定性评分"),
        "毛利率均值": fundamental_snapshot.get("毛利率均值"),
        "毛利率波动": fundamental_snapshot.get("毛利率波动"),
        "毛利率稳定性评分": fundamental_snapshot.get("毛利率稳定性评分"),
        "经营现金流净利润比": fundamental_snapshot.get("经营现金流净利润比"),
        "资产负债率": fundamental_snapshot.get("资产负债率"),
        "入选理由": "；".join(reasons),
        "风险提示": "；".join(risks)
    })

    record.pop("_technical_reasons", None)
    record.pop("_technical_risks", None)

    return record


def screen_stock_pool(
    symbols: list[str],
    days: int = 120,
    top_n: int = 10,
    technical_weight: float = 0.6,
    fundamental_weight: float = 0.4,
    fundamental_review_limit: int = 20
) -> pd.DataFrame:
    technical_results = []

    for symbol in symbols:
        try:
            technical_results.append(
                analyze_technical_for_screening(symbol=symbol, days=days)
            )
        except Exception as error:
            technical_results.append({
                "股票代码": symbol,
                "综合评分": 0,
                "技术评分": 0,
                "基本面评分": 0,
                "技术等级": "错误",
                "基本面等级": "待复评",
                "最新收盘价": None,
                "MA20": None,
                "RSI14": None,
                "MACD": None,
                "财报日期": None,
                "ROE": None,
                "净利润同比": None,
                "净利率": None,
                "毛利率": None,
                "净利润": None,
                "主营收入": None,
                "营收同比": None,
                "ROE均值": None,
                "ROE波动": None,
                "ROE稳定性评分": None,
                "毛利率均值": None,
                "毛利率波动": None,
                "毛利率稳定性评分": None,
                "经营现金流净利润比": None,
                "资产负债率": None,
                "_technical_reasons": [],
                "_technical_risks": [str(error)]
            })

    technical_df = pd.DataFrame(technical_results)

    if technical_df.empty:
        return technical_df

    technical_df = technical_df.sort_values(
        by="技术评分",
        ascending=False
    )

    if fundamental_weight <= 0:
        result_df = technical_df.head(top_n).copy()
        result_df["入选理由"] = result_df["_technical_reasons"].map(lambda items: "；".join(items))
        result_df["风险提示"] = result_df["_technical_risks"].map(lambda items: "；".join(items))
        return result_df.drop(columns=["_technical_reasons", "_technical_risks"])

    review_limit = max(top_n, fundamental_review_limit)
    review_limit = min(review_limit, len(technical_df))

    review_records = technical_df.head(review_limit).to_dict(orient="records")
    reviewed_records = [
        attach_fundamental_review(
            record=record,
            technical_weight=technical_weight,
            fundamental_weight=fundamental_weight
        )
        for record in review_records
    ]

    result_df = pd.DataFrame(reviewed_records)

    result_df = result_df.sort_values(
        by="综合评分",
        ascending=False
    )

    return result_df.head(top_n)
