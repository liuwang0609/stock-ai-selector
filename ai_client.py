import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_ai_config() -> dict:
    api_key = (
        os.getenv("AI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("AI_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or None
    )
    model = (
        os.getenv("AI_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "deepseek-v4-flash"
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model
    }


def has_ai_api_key() -> bool:
    config = get_ai_config()
    return bool(config["api_key"])


def has_openai_api_key() -> bool:
    return has_ai_api_key()


def generate_ai_analysis(prompt: str) -> str:
    config = get_ai_config()

    if not config["api_key"]:
        return (
            "当前未配置 AI API Key，因此这里显示的是 AI 分析占位内容。\n\n"
            "配置 AI_API_KEY、AI_BASE_URL 和 AI_MODEL 后，这里会调用模型，"
            "根据批量筛选结果生成最终 10 只股票的分析理由、风险提示和排序说明。"
        )

    client_kwargs = {"api_key": config["api_key"]}

    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]

    client = OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个严谨的A股研究报告助手。"
                        "你只基于用户提供的数据进行解释，不编造新闻、公告、财报或未来涨跌。"
                        "你的输出仅用于学习和研究，不构成投资建议。"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as error:
        error_text = str(error)

        if "Insufficient Balance" in error_text or "402" in error_text:
            return (
                "AI API 调用失败：当前 API 账号余额不足。\n\n"
                "这说明 API Key、Base URL 和模型配置大概率是正确的，但账户没有可用额度。"
                "请到对应 API 平台充值或领取额度后再试。"
            )

        return f"AI API 调用失败：{error_text}"
