import pandas as pd
import streamlit as st

from ai_client import generate_ai_analysis
from data_loader import get_stock_history, to_baostock_code
from fundamentals import get_fundamental_snapshot, score_fundamentals
from indicators import add_technical_indicators
from industry_scoring import (
    add_industry_competitiveness,
    build_industry_activity_summary,
    filter_top_industries
)
from liquidity_filter import rank_stocks_by_latest_volume
from prompt_builder import build_stock_selection_prompt
from screener import DEFAULT_STOCK_POOL, score_stock, screen_stock_pool
from stock_universe import get_a_stock_universe


@st.cache_data(ttl=3600)
def load_stock_universe(include_bj: bool) -> pd.DataFrame:
    return get_a_stock_universe(include_bj=include_bj)


@st.cache_data(ttl=86400)
def load_top_volume_pool(stock_pool_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return rank_stocks_by_latest_volume(
        stock_pool_df=stock_pool_df,
        top_n=top_n
    )


def format_number(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def apply_app_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f7f9;
            color: #172033;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #eef1f5;
            border-right: 1px solid #d8dde6;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #172033;
        }

        .app-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            background: #ffffff;
        }

        .app-header h1 {
            margin: 0;
            color: #111827;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 750;
            letter-spacing: 0;
        }

        .app-header p {
            margin: 0.45rem 0 0 0;
            color: #586174;
            font-size: 0.98rem;
        }

        .app-badge {
            border: 1px solid #b9c5d8;
            border-radius: 999px;
            color: #2f4f4f;
            background: #f8fafc;
            font-size: 0.86rem;
            padding: 0.35rem 0.75rem;
            white-space: nowrap;
        }

        h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-left: 4px solid #2f6f5e;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: #5f697a;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 750;
        }

        .stButton > button {
            background: #2f6f5e;
            color: white;
            border: 1px solid #2f6f5e;
            border-radius: 6px;
            font-weight: 700;
            min-height: 2.45rem;
        }

        .stButton > button:hover {
            background: #265a4d;
            border-color: #265a4d;
            color: white;
        }

        .stDownloadButton > button {
            background: #ffffff;
            color: #2f6f5e;
            border: 1px solid #2f6f5e;
            border-radius: 6px;
            font-weight: 700;
            min-height: 2.45rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #d8dde6;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 0.55rem 0.85rem;
            background: #ffffff;
            border: 1px solid #d8dde6;
        }

        .stTabs [aria-selected="true"] {
            border-color: #2f6f5e;
            color: #2f6f5e;
            font-weight: 700;
        }

        hr {
            border: none;
            border-top: 1px solid #d8dde6;
            margin: 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_app_header():
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1>A股AI选股助手</h1>
                <p>技术面筛选 · 基本面复评 · 成交量预筛 · AI研究报告</p>
            </div>
            <div class="app-badge">研究辅助工具</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def attach_stock_metadata(result_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty or metadata_df.empty:
        return result_df

    metadata_columns = ["股票代码", "股票名称", "市场板块", "行业板块", "行业分类"]
    available_columns = [
        column
        for column in metadata_columns
        if column in metadata_df.columns
    ]

    merged_df = result_df.merge(
        metadata_df[available_columns].drop_duplicates("股票代码"),
        on="股票代码",
        how="left"
    )

    front_columns = [
        column
        for column in ["股票代码", "股票名称", "市场板块", "行业板块", "行业分类"]
        if column in merged_df.columns
    ]
    other_columns = [
        column
        for column in merged_df.columns
        if column not in front_columns
    ]

    return merged_df[front_columns + other_columns]


def get_stock_metadata(symbol: str) -> dict:
    baostock_code = to_baostock_code(symbol)
    stock_code = baostock_code.split(".")[-1]

    try:
        universe_df = load_stock_universe(include_bj=True)
        matched_df = universe_df[universe_df["股票代码"] == stock_code]

        if not matched_df.empty:
            return matched_df.iloc[0].to_dict()

    except Exception:
        pass

    return {
        "股票代码": stock_code,
        "股票名称": "未知",
        "市场板块": "未知",
        "行业板块": "未知",
        "行业分类": "未知"
    }


def render_single_stock_analysis():
    st.sidebar.header("单只股票参数")

    symbol = st.sidebar.text_input("股票代码", value="000001")
    days = st.sidebar.slider("显示最近多少个交易日", 60, 250, 120)

    if st.button("获取股票数据"):
        with st.spinner("正在获取数据并计算技术指标..."):
            try:
                df = get_stock_history(symbol=symbol, days=days)

                if df.empty:
                    st.warning("没有获取到数据，请检查股票代码。")
                    return

                df = add_technical_indicators(df)
                latest = df.iloc[-1]
                analysis = score_stock(df)
                fundamental_snapshot = get_fundamental_snapshot(symbol)
                fundamental_analysis = score_fundamentals(fundamental_snapshot)

                metadata = get_stock_metadata(symbol)
                stock_title = f"{metadata['股票代码']} {metadata['股票名称']}"

                st.subheader(f"{stock_title} 基本信息")

                info_df = pd.DataFrame([{
                    "股票代码": metadata["股票代码"],
                    "股票名称": metadata["股票名称"],
                    "市场板块": metadata["市场板块"],
                    "行业板块": metadata["行业板块"],
                    "行业分类": metadata["行业分类"]
                }])
                st.dataframe(info_df, use_container_width=True, hide_index=True)

                st.subheader(f"{stock_title} 技术面概览")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("最新收盘价", format_number(latest["收盘"]))
                col2.metric("MA20", format_number(latest["MA20"]))
                col3.metric("RSI14", format_number(latest["RSI14"]))
                col4.metric("MACD", format_number(latest["MACD"], 4))

                st.subheader("技术面打分")

                score_col1, score_col2 = st.columns(2)
                score_col1.metric("综合评分", f"{analysis['score']} / 100")
                score_col2.metric("技术等级", analysis["grade"])

                if analysis["reasons"]:
                    st.success("入选理由")
                    for reason in analysis["reasons"]:
                        st.write(f"- {reason}")

                if analysis["risks"]:
                    st.warning("风险提示")
                    for risk in analysis["risks"]:
                        st.write(f"- {risk}")

                st.subheader("基本面概览")

                fundamental_col1, fundamental_col2, fundamental_col3, fundamental_col4 = st.columns(4)
                fundamental_col1.metric("基本面评分", f"{fundamental_analysis['score']} / 100")
                fundamental_col2.metric("ROE", format_number(fundamental_snapshot.get("ROE")))
                fundamental_col3.metric("ROE稳定性", format_number(fundamental_snapshot.get("ROE稳定性评分")))
                fundamental_col4.metric("净利润同比", format_number(fundamental_snapshot.get("净利润同比")))

                growth_col1, growth_col2, growth_col3, growth_col4 = st.columns(4)
                growth_col1.metric("营收同比", format_number(fundamental_snapshot.get("营收同比")))
                growth_col2.metric("净利率", format_number(fundamental_snapshot.get("净利率")))
                growth_col3.metric("毛利率", format_number(fundamental_snapshot.get("毛利率")))
                growth_col4.metric("毛利率稳定性", format_number(fundamental_snapshot.get("毛利率稳定性评分")))

                st.caption(f"使用财报日期：{fundamental_snapshot.get('财报日期', 'N/A')}")

                if fundamental_analysis["reasons"]:
                    st.success("基本面优势")
                    for reason in fundamental_analysis["reasons"]:
                        st.write(f"- {reason}")

                if fundamental_analysis["risks"]:
                    st.warning("基本面风险")
                    for risk in fundamental_analysis["risks"]:
                        st.write(f"- {risk}")

                st.subheader("价格与均线图")

                chart_tab1, chart_tab2 = st.tabs(["趋势线", "均线偏离率"])

                with chart_tab1:
                    trend_chart_df = df.set_index("日期")[["收盘", "MA5", "MA10", "MA20", "MA60"]]
                    st.line_chart(trend_chart_df)
                    st.caption("趋势线展示收盘价和 MA5、MA10、MA20、MA60；如果线条较密，可以切到“均线偏离率”查看差异。")

                with chart_tab2:
                    spread_df = df.copy()
                    spread_df["收盘价相对MA20偏离率(%)"] = (spread_df["收盘"] / spread_df["MA20"] - 1) * 100
                    spread_df["MA5相对MA20偏离率(%)"] = (spread_df["MA5"] / spread_df["MA20"] - 1) * 100
                    spread_df["MA20相对MA60偏离率(%)"] = (spread_df["MA20"] / spread_df["MA60"] - 1) * 100

                    spread_chart_df = spread_df.set_index("日期")[[
                        "收盘价相对MA20偏离率(%)",
                        "MA5相对MA20偏离率(%)",
                        "MA20相对MA60偏离率(%)"
                    ]]
                    st.line_chart(spread_chart_df)
                    st.caption("偏离率图用百分比展示均线之间的距离，更容易看出短期和中期趋势变化。")

                st.subheader("最近交易日数据")

                display_columns = [
                    "日期", "股票代码", "开盘", "最高", "最低", "收盘",
                    "成交量", "成交额", "换手率", "涨跌幅",
                    "MA5", "MA10", "MA20", "MA60",
                    "RSI14", "MACD_DIF", "MACD_DEA", "MACD"
                ]

                st.dataframe(df[display_columns].tail(30), use_container_width=True)

            except Exception as error:
                st.error("获取数据或计算指标失败。")
                st.exception(error)


def render_manual_pool_editor() -> tuple[list[str], pd.DataFrame]:
    stock_text = st.text_area(
        "股票池，每行一个股票代码",
        value="\n".join(DEFAULT_STOCK_POOL),
        height=240
    )

    symbols = [
        item.strip()
        for item in stock_text.replace(",", "\n").splitlines()
        if item.strip()
    ]

    try:
        universe_df = load_stock_universe(include_bj=True)
        metadata_df = universe_df[universe_df["股票代码"].isin(symbols)].copy()
    except Exception:
        metadata_df = pd.DataFrame({"股票代码": symbols})

    st.subheader("当前股票池预览")
    st.dataframe(metadata_df, use_container_width=True)

    return symbols, metadata_df


def render_full_market_pool_editor() -> tuple[list[str], pd.DataFrame]:
    include_bj = st.checkbox("包含北交所股票", value=True)

    with st.spinner("正在加载全 A 股股票池..."):
        universe_df = load_stock_universe(include_bj=include_bj)

    st.metric("全 A 股股票池数量", len(universe_df))

    board_options = sorted(universe_df["市场板块"].dropna().unique().tolist())
    selected_boards = st.multiselect(
        "按市场板块过滤",
        board_options,
        default=board_options
    )

    filtered_df = universe_df.copy()

    if selected_boards:
        filtered_df = filtered_df[filtered_df["市场板块"].isin(selected_boards)].copy()

    industry_options = sorted(filtered_df["行业板块"].dropna().unique().tolist())
    selected_industries = st.multiselect(
        "按行业板块过滤（不选则不过滤）",
        industry_options
    )

    if selected_industries:
        filtered_df = filtered_df[filtered_df["行业板块"].isin(selected_industries)].copy()

    st.subheader("股票池预览")
    st.caption("预览最多显示前 500 行。筛选时使用下方“本次实际扫描数量”控制运行时间。")
    st.dataframe(filtered_df.head(500), use_container_width=True)

    if filtered_df.empty:
        return [], filtered_df

    scan_all = st.checkbox("扫描筛选后的全部股票（可能很慢）", value=False)

    if scan_all:
        st.session_state["full_market_selection_method"] = "全部股票"
        st.session_state["full_market_filtered_pool"] = filtered_df
        st.session_state["full_market_scan_count"] = len(filtered_df)
        scan_df = filtered_df
    else:
        default_scan_count = min(200, len(filtered_df))
        scan_count = st.number_input(
            "本次实际扫描数量",
            min_value=1,
            max_value=len(filtered_df),
            value=default_scan_count,
            step=50
        )
        st.session_state["full_market_selection_method"] = "按最近交易日成交量前N只"
        st.session_state["full_market_filtered_pool"] = filtered_df
        st.session_state["full_market_scan_count"] = int(scan_count)
        scan_df = filtered_df

    if scan_all:
        st.info(f"本次将扫描筛选后的全部 {len(scan_df)} 只股票。")
    else:
        st.info(f"点击开始后，会先按最近交易日成交量排序，再取成交量前 {int(scan_count)} 只进入评分模型。")

    symbols = scan_df["股票代码"].tolist()

    return symbols, universe_df


def render_batch_screening():
    st.subheader("批量筛选股票")
    st.caption("全市场扫描会逐只获取日线数据，运行时间较长；课堂演示建议先限制扫描数量。")

    pool_mode = st.radio(
        "股票池来源",
        ["全 A 股市场", "手动输入测试股票池"],
        horizontal=True
    )

    if pool_mode == "全 A 股市场":
        symbols, metadata_df = render_full_market_pool_editor()
    else:
        symbols, metadata_df = render_manual_pool_editor()

    days = st.slider("每只股票使用最近多少个交易日", 80, 250, 120)
    top_n = st.slider("最终显示前多少只", 5, 20, 10)
    industry_first = st.checkbox("启用行业优先筛选", value=True)
    top_industry_count = st.slider("优先行业数量", 1, 20, 5)
    industry_competitiveness_weight_percent = st.slider("行业内竞争力权重", 0, 50, 20, step=5)
    technical_weight_percent = st.slider("技术面权重", 0, 100, 60, step=5)
    technical_weight = technical_weight_percent / 100
    fundamental_weight = 1 - technical_weight
    fundamental_review_limit = st.slider(
        "基本面复评数量",
        min_value=top_n,
        max_value=100,
        value=top_n,
        step=5
    )

    st.caption(
        f"当前综合评分 = 技术评分 {technical_weight_percent}% + "
        f"基本面评分 {int(fundamental_weight * 100)}%；"
        f"先技术初筛，再对技术排名前 {fundamental_review_limit} 只做基本面复评。"
    )

    if industry_first:
        st.caption(
            f"行业优先已开启：先选成交活跃度靠前的 {top_industry_count} 个行业；"
            f"最终排序额外加入 {industry_competitiveness_weight_percent}% 的行业内竞争力评分。"
        )

    if st.button("开始批量筛选"):
        if not symbols:
            st.warning("当前股票池为空，请调整股票池或过滤条件。")
            return

        if (
            pool_mode == "全 A 股市场"
            and st.session_state.get("full_market_selection_method") == "按最近交易日成交量前N只"
        ):
            filtered_pool_df = st.session_state.get("full_market_filtered_pool", pd.DataFrame())
            scan_count = st.session_state.get("full_market_scan_count", 200)

            if filtered_pool_df.empty:
                st.warning("当前过滤后的股票池为空，无法进行成交量排序。")
                return

            with st.spinner("正在按最近交易日成交量排序，并选出成交量前 N 只股票..."):
                ranked_volume_df = load_top_volume_pool(
                    stock_pool_df=filtered_pool_df,
                    top_n=int(scan_count)
                )

            if ranked_volume_df.empty:
                st.warning("没有获取到成交量排序结果。")
                return

            st.subheader("本次实际进入评分模型的成交量前 N 股票池")
            st.dataframe(ranked_volume_df, use_container_width=True)

            if industry_first:
                industry_summary_df = build_industry_activity_summary(ranked_volume_df)

                if not industry_summary_df.empty:
                    st.subheader("优先行业")
                    st.dataframe(
                        industry_summary_df.head(top_industry_count),
                        use_container_width=True
                    )

                    ranked_volume_df = filter_top_industries(
                        stock_pool_df=ranked_volume_df,
                        top_industry_count=top_industry_count
                    )

                    st.subheader("行业优先筛选后的股票池")
                    st.dataframe(ranked_volume_df, use_container_width=True)

            symbols = ranked_volume_df["股票代码"].tolist()
            metadata_df = ranked_volume_df

        model_result_count = top_n
        review_limit_for_model = fundamental_review_limit

        if industry_first:
            model_result_count = min(len(symbols), max(top_n, top_n * 2))
            review_limit_for_model = max(fundamental_review_limit, model_result_count)

        with st.spinner("正在批量获取数据并计算评分，可能需要一段时间..."):
            result_df = screen_stock_pool(
                symbols=symbols,
                days=days,
                top_n=model_result_count,
                technical_weight=technical_weight,
                fundamental_weight=fundamental_weight,
                fundamental_review_limit=review_limit_for_model
            )

        if result_df.empty:
            st.warning("没有筛选结果。")
            return

        result_df = attach_stock_metadata(result_df, metadata_df)

        if industry_first:
            result_df = add_industry_competitiveness(result_df)
            result_df["原模型评分"] = result_df["综合评分"]
            industry_weight = industry_competitiveness_weight_percent / 100
            result_df["综合评分"] = (
                result_df["原模型评分"] * (1 - industry_weight)
                + result_df["行业内竞争力评分"] * industry_weight
            ).round(2)
            result_df = result_df.sort_values(
                by="综合评分",
                ascending=False
            )

        result_df = result_df.head(top_n)

        st.subheader("筛选结果")
        st.dataframe(result_df, use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="下载筛选结果 CSV",
            data=csv_data,
            file_name="stock_screening_result.csv",
            mime="text/csv"
        )

        prompt = build_stock_selection_prompt(result_df)

        with st.expander("查看发送给 AI 的提示词"):
            st.code(prompt, language="text")

        st.subheader("AI 最终分析")

        ai_result = generate_ai_analysis(prompt)
        st.markdown(ai_result)


st.set_page_config(
    page_title="A股AI选股助手",
    layout="wide"
)

apply_app_style()
render_app_header()
st.caption("仅供学习和研究，不构成投资建议。")

st.sidebar.header("功能选择")

mode = st.sidebar.radio(
    "选择功能",
    ["单只股票分析", "批量筛选前10"]
)

if mode == "单只股票分析":
    render_single_stock_analysis()
else:
    render_batch_screening()
