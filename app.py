from html import escape

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


def render_section_heading(title: str, caption: str | None = None):
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="section-heading"><h2>{escape(title)}</h2>{caption_html}</div>',
        unsafe_allow_html=True
    )


def render_workflow_strip():
    steps = [
        ("01", "行业优先"),
        ("02", "成交活跃"),
        ("03", "质量复评"),
        ("04", "技术择时"),
        ("05", "AI报告")
    ]
    step_html = "".join(
        f'<div class="workflow-step"><span class="workflow-index">{index}</span><span>{label}</span></div>'
        for index, label in steps
    )
    st.markdown(f'<div class="workflow-strip">{step_html}</div>', unsafe_allow_html=True)


def render_setting_summary(items: list[tuple[str, str]]):
    item_html = "".join(
        f'<span class="setting-pill"><span class="setting-label">{escape(label)}</span><span class="setting-value">{escape(value)}</span></span>'
        for label, value in items
    )
    st.markdown(f'<div class="setting-summary">{item_html}</div>', unsafe_allow_html=True)


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

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.1rem;
        }

        .app-header {
            align-items: center;
            padding: 1.25rem 1.35rem;
            border-color: #d7dde7;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.06);
        }

        .app-header h1 {
            font-size: 1.9rem;
            color: #0f172a;
        }

        .app-header p {
            color: #5b6472;
        }

        .app-header-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .header-chip,
        .app-badge {
            border: 1px solid #d1d9e6;
            border-radius: 999px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.82rem;
            line-height: 1;
            padding: 0.45rem 0.7rem;
            white-space: nowrap;
        }

        .app-badge {
            border-color: #b7d4ca;
            background: #eef8f3;
            color: #236052;
            font-weight: 700;
        }

        .workflow-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0 0 1.1rem 0;
        }

        .workflow-step {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-height: 3rem;
            padding: 0.65rem 0.75rem;
            border: 1px solid #d7dde7;
            border-radius: 8px;
            background: #ffffff;
            color: #273449;
            font-weight: 700;
            box-shadow: 0 1px 2px rgba(31, 41, 55, 0.04);
        }

        .workflow-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 999px;
            background: #eaf2ff;
            color: #255bb8;
            font-size: 0.78rem;
            flex: 0 0 auto;
        }

        .section-heading {
            margin: 1.25rem 0 0.75rem;
            padding: 0 0 0.15rem;
        }

        .section-heading h2 {
            margin: 0;
            padding-left: 0.65rem;
            border-left: 4px solid #2f6f5e;
            color: #111827;
            font-size: 1.18rem;
            line-height: 1.35;
            font-weight: 780;
            letter-spacing: 0;
        }

        .section-heading p {
            margin: 0.35rem 0 0 0.9rem;
            color: #667085;
            font-size: 0.92rem;
        }

        .setting-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.85rem 0 0.35rem;
            padding: 0.8rem;
            border: 1px solid #d7dde7;
            border-left: 4px solid #255bb8;
            border-radius: 8px;
            background: #ffffff;
        }

        .setting-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            min-height: 2rem;
            padding: 0.35rem 0.65rem;
            border: 1px solid #e0e6ef;
            border-radius: 999px;
            background: #f8fafc;
            white-space: nowrap;
        }

        .setting-label {
            color: #667085;
            font-size: 0.82rem;
        }

        .setting-value {
            color: #172033;
            font-weight: 750;
            font-size: 0.88rem;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
        }

        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stSlider {
            padding-bottom: 0.35rem;
        }

        [data-testid="stWidgetLabel"] p {
            color: #344054;
            font-weight: 700;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            border-left-color: #2f6f5e;
            box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
        }

        div[data-testid="stExpander"] {
            border: 1px solid #d7dde7;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
        }

        div[data-testid="stExpander"] details summary {
            font-weight: 750;
            color: #172033;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus {
            box-shadow: 0 0 0 3px rgba(47, 111, 94, 0.18);
        }

        .stRadio [role="radiogroup"] {
            gap: 0.5rem;
        }

        div[data-testid="stDataFrame"] {
            box-shadow: 0 4px 14px rgba(31, 41, 55, 0.05);
        }

        @media (max-width: 900px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .workflow-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
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
                <p>行业优先 · 基本面复评 · 技术面择时 · AI研究报告</p>
                <div class="app-header-meta">
                    <span class="header-chip">全A股股票池</span>
                    <span class="header-chip">行业优先筛选</span>
                    <span class="header-chip">规则评分 + AI总结</span>
                </div>
            </div>
            <div class="app-badge">研究辅助工具</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_scoring_logic_explainer():
    with st.expander("评分逻辑与行业量化方法", expanded=False):
        st.markdown(
            """
            系统不是让 AI 直接选股，而是先用规则模型量化评分，再由 AI 生成研究报告。
            单只股票分析页展示的是个股技术面和基本面；批量筛选页会进一步加入行业优先和行业内竞争力评分。
            """
        )

        formula_df = pd.DataFrame([
            {
                "模块": "技术面评分",
                "作用": "判断趋势、动量、量能和回撤是否健康",
                "主要指标": "MA5、MA10、MA20、MA60、RSI14、MACD、成交量MA、近20日回撤"
            },
            {
                "模块": "基本面评分",
                "作用": "判断公司盈利质量、成长性、现金流和财务风险",
                "主要指标": "ROE、ROE稳定性、净利润同比、营收同比、净利率、毛利率稳定性、现金流、资产负债率、EPS"
            },
            {
                "模块": "行业优先评分",
                "作用": "先看哪些行业更活跃、更值得进入候选池",
                "主要指标": "行业平均涨跌幅、行业上涨占比、行业中位成交额"
            },
            {
                "模块": "行业内竞争力评分",
                "作用": "判断公司在所属行业里的相对地位",
                "主要指标": "营收行业分位、净利润行业分位、ROE行业分位、毛利率行业分位、净利率行业分位、营收增速行业分位、净利润增速行业分位"
            },
            {
                "模块": "技术/壁垒代理评分",
                "作用": "用可量化指标近似衡量公司是否具备产品、技术、成本或品牌壁垒",
                "主要指标": "毛利率行业分位、ROE行业分位、ROE稳定性评分、毛利率稳定性评分"
            }
        ])
        st.dataframe(formula_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            批量筛选中的综合排序逻辑：

            1. 先从全 A 股中按行业计算中位成交额、平均涨跌幅和上涨占比，选出优先行业。
            2. 在优先行业中选择成交量和成交额更活跃的股票进入候选池。
            3. 对候选股票计算基本面评分和行业内竞争力评分。
            4. 对候选股票计算技术评分，主要用于择时确认。
            5. 最终综合评分会在原模型评分基础上，额外加入行业内竞争力权重。

            当前默认逻辑可以概括为：
            """
        )

        st.code(
            "原模型评分 = 技术评分 * 技术面权重 + 基本面评分 * 基本面权重\n"
            "最终综合评分 = 原模型评分 * (1 - 行业内竞争力权重) + 行业内竞争力评分 * 行业内竞争力权重",
            language="text"
        )

        industry_df = pd.DataFrame([
            {
                "概念": "行业活跃度",
                "量化方式": "优先看行业平均涨跌幅、上涨占比和中位成交额，弱化总成交额带来的体量偏差。"
            },
            {
                "概念": "行业地位",
                "量化方式": "在同一行业内比较 ROE、毛利率、净利率、营收/净利润行业分位和综合评分，规模只做辅助，不作为主导。"
            },
            {
                "概念": "话语权/定价权",
                "量化方式": "用毛利率、净利率、ROE 高于同行的程度做代理指标。"
            },
            {
                "概念": "ROE稳定性",
                "量化方式": "计算最近多期 ROE 的均值和波动，均值越高、波动越小，稳定性越好。"
            },
            {
                "概念": "细分龙头",
                "量化方式": "行业内竞争力评分高的公司标记为“强”，中等为“中”，其余为“观察”。"
            }
        ])
        st.dataframe(industry_df, use_container_width=True, hide_index=True)


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

                render_section_heading(f"{stock_title} 基本信息", "代码、名称、市场板块和行业信息")

                info_df = pd.DataFrame([{
                    "股票代码": metadata["股票代码"],
                    "股票名称": metadata["股票名称"],
                    "市场板块": metadata["市场板块"],
                    "行业板块": metadata["行业板块"],
                    "行业分类": metadata["行业分类"]
                }])
                st.dataframe(info_df, use_container_width=True, hide_index=True)

                render_scoring_logic_explainer()

                render_section_heading(f"{stock_title} 技术面概览", "价格、均线、动量和短期趋势")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("最新收盘价", format_number(latest["收盘"]))
                col2.metric("MA20", format_number(latest["MA20"]))
                col3.metric("RSI14", format_number(latest["RSI14"]))
                col4.metric("MACD", format_number(latest["MACD"], 4))

                render_section_heading("技术面打分", "用均线、RSI、MACD、量能和回撤判断当前走势")

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

                render_section_heading("基本面概览", "用盈利能力、成长性、现金流和负债水平判断公司质量")

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

                render_section_heading("价格与均线图", "趋势线看方向，偏离率看均线之间的距离")

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

                render_section_heading("最近交易日数据")

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

    render_section_heading("当前股票池预览")
    st.dataframe(metadata_df, use_container_width=True)

    return symbols, metadata_df


def render_full_market_pool_editor() -> tuple[list[str], pd.DataFrame]:
    include_bj = True

    with st.spinner("正在加载全 A 股股票池..."):
        universe_df = load_stock_universe(include_bj=include_bj)

    st.metric("全 A 股股票池数量", len(universe_df))

    filtered_df = universe_df.copy()

    with st.expander("股票池过滤（可选）", expanded=False):
        include_bj = st.checkbox("包含北交所股票", value=True)

        if include_bj != True:
            with st.spinner("正在重新加载股票池..."):
                universe_df = load_stock_universe(include_bj=include_bj)
            filtered_df = universe_df.copy()

        board_options = sorted(universe_df["市场板块"].dropna().unique().tolist())
        selected_boards = st.multiselect(
            "市场板块",
            board_options,
            default=board_options
        )

        if selected_boards:
            filtered_df = filtered_df[filtered_df["市场板块"].isin(selected_boards)].copy()

        industry_options = sorted(filtered_df["行业板块"].dropna().unique().tolist())
        selected_industries = st.multiselect(
            "行业板块",
            industry_options
        )

        if selected_industries:
            filtered_df = filtered_df[filtered_df["行业板块"].isin(selected_industries)].copy()

    render_section_heading("股票池预览", "默认先看行业，再从优先行业中选择成交量活跃的股票，预览最多显示前 300 行")
    st.dataframe(filtered_df.head(300), use_container_width=True)

    if filtered_df.empty:
        return [], filtered_df

    default_scan_count = min(50, len(filtered_df))
    scan_col1, scan_col2 = st.columns([1, 2])

    with scan_col1:
        scan_count = st.number_input(
            "扫描成交量前多少只股票",
            min_value=10,
            max_value=len(filtered_df),
            value=default_scan_count,
            step=10
        )

    with scan_col2:
        st.caption("建议先用默认数量测试，确认网页响应正常后再扩大扫描范围。")

    with st.expander("运行范围（高级）", expanded=False):
        scan_all = st.checkbox("扫描筛选后的全部股票（非常慢）", value=False)

    if scan_all:
        st.session_state["full_market_selection_method"] = "全部股票"
        st.session_state["full_market_filtered_pool"] = filtered_df
        st.session_state["full_market_scan_count"] = len(filtered_df)
        scan_df = filtered_df
    else:
        st.session_state["full_market_selection_method"] = "按最近交易日成交量前N只"
        st.session_state["full_market_filtered_pool"] = filtered_df
        st.session_state["full_market_scan_count"] = int(scan_count)
        scan_df = filtered_df

    if scan_all:
        st.info(f"本次将扫描筛选后的全部 {len(scan_df)} 只股票。")
    else:
        st.info(f"点击开始后，会先选择优先行业，再从优先行业中取成交量前 {int(scan_count)} 只进入评分模型。")

    symbols = scan_df["股票代码"].tolist()

    return symbols, universe_df


def render_batch_screening():
    render_section_heading("批量筛选股票", "默认流程：优先行业 → 行业内基本面与成交量筛选 → 技术面择时确认 → AI研究报告")

    pool_mode = st.radio(
        "股票池来源",
        ["全 A 股市场", "手动输入测试股票池"],
        horizontal=True
    )

    if pool_mode == "全 A 股市场":
        symbols, metadata_df = render_full_market_pool_editor()
    else:
        symbols, metadata_df = render_manual_pool_editor()

    setting_col1, setting_col2, setting_col3 = st.columns([1, 1, 1])

    with setting_col1:
        top_n = st.number_input(
            "最终输出股票数量",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )

    with setting_col2:
        industry_first = st.checkbox("启用行业优先模型", value=True)

    with setting_col3:
        industry_group_column = st.selectbox(
            "板块评分口径",
            ["行业板块", "市场板块"],
            index=0
        )

    days = 120
    top_industry_count = 5
    industry_competitiveness_weight_percent = 20
    technical_weight_percent = 60
    fundamental_review_limit = int(top_n)

    with st.expander("高级评分参数", expanded=False):
        days = st.number_input(
            "日线数据观察天数",
            min_value=80,
            max_value=250,
            value=120,
            step=10
        )
        top_industry_count = st.number_input(
            "优先行业数量",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        industry_competitiveness_weight_percent = st.slider(
            "行业内竞争力权重",
            0,
            50,
            20,
            step=5
        )
        technical_weight_percent = st.slider(
            "技术面权重",
            0,
            100,
            60,
            step=5
        )
        fundamental_review_limit = st.number_input(
            "基本面复评数量",
            min_value=int(top_n),
            max_value=100,
            value=int(top_n),
            step=5
        )

    technical_weight = technical_weight_percent / 100
    fundamental_weight = 1 - technical_weight

    render_setting_summary([
        ("输出", f"{int(top_n)} 只"),
        ("技术 / 基本面", f"{technical_weight_percent}% / {int(fundamental_weight * 100)}%"),
        ("基本面复评", f"{int(fundamental_review_limit)} 只"),
        ("行业竞争力", f"{industry_competitiveness_weight_percent}%")
    ])

    if industry_first:
        st.caption(
            f"行业优先已开启：按“{industry_group_column}”选成交活跃度靠前的 {int(top_industry_count)} 个板块；"
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

            if industry_first:
                with st.spinner("正在计算行业活跃度，并选出优先行业..."):
                    market_snapshot_df = load_top_volume_pool(
                        stock_pool_df=filtered_pool_df,
                        top_n=len(filtered_pool_df)
                    )

                if market_snapshot_df.empty:
                    st.warning("没有获取到行业快照数据。")
                    return

                industry_summary_df = build_industry_activity_summary(
                    market_snapshot_df,
                    group_column=industry_group_column
                )

                if not industry_summary_df.empty:
                    render_section_heading("优先行业", "按行业活跃度评分排序")
                    st.dataframe(
                        industry_summary_df.head(top_industry_count),
                        use_container_width=True
                    )

                    industry_pool_df = filter_top_industries(
                        stock_pool_df=market_snapshot_df,
                        top_industry_count=top_industry_count,
                        group_column=industry_group_column
                    )

                    ranked_volume_df = (
                        industry_pool_df
                        .sort_values("当日成交量", ascending=False)
                        .head(int(scan_count))
                        .reset_index(drop=True)
                    )
                else:
                    ranked_volume_df = (
                        market_snapshot_df
                        .sort_values("当日成交量", ascending=False)
                        .head(int(scan_count))
                        .reset_index(drop=True)
                    )

                render_section_heading("优先行业中的成交量活跃股票", "从优先行业里按当天成交量排序")
                st.dataframe(ranked_volume_df, use_container_width=True)

            else:
                with st.spinner("正在按最近交易日成交量排序，并选出成交量前 N 只股票..."):
                    ranked_volume_df = load_top_volume_pool(
                        stock_pool_df=filtered_pool_df,
                        top_n=int(scan_count)
                    )

                if ranked_volume_df.empty:
                    st.warning("没有获取到成交量排序结果。")
                    return

                render_section_heading("本次实际进入评分模型的成交量前 N 股票池")
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

        render_section_heading("筛选结果", "按最终综合评分排序输出")
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

        render_section_heading("AI 最终分析", "基于筛选结果生成研究报告和风险提示")

        ai_result = generate_ai_analysis(prompt)
        st.markdown(ai_result)


st.set_page_config(
    page_title="A股AI选股助手",
    layout="wide"
)

apply_app_style()
render_app_header()
render_workflow_strip()
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
