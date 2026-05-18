from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.env_loader import load_env_file
from src.llm_responses_client import ResponsesClient, response_to_pretty_json
from src.prompt_security import SYSTEM_INSTRUCTIONS, build_user_prompt, find_prompt_injection


APP_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = APP_DIR / ".env"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
ALLOWED_MEMORY_LIMITS = {"1g", "4g", "16g", "64g"}


def configure_page() -> None:
    st.set_page_config(
        page_title="LLM-аналитик CSV",
        layout="wide",
    )


def load_configuration() -> dict[str, str | int]:
    load_env_file(DEFAULT_ENV_FILE)
    max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "20"))
    memory_limit = os.environ.get("CODE_INTERPRETER_MEMORY", "1g")
    if memory_limit not in ALLOWED_MEMORY_LIMITS:
        memory_limit = "1g"
    return {
        "api_key": os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "provider_name": os.environ.get("LLM_PROVIDER_NAME", "openai"),
        "memory_limit": memory_limit,
        "max_upload_mb": max_upload_mb,
    }


def render_sidebar(config: dict[str, str | int]) -> dict[str, str | int]:
    with st.sidebar:
        st.header("Настройки LLM")
        api_key = st.text_input(
            "LLM_API_KEY",
            value=str(config["api_key"]),
            type="password",
            help="Можно указать здесь или в work-3/.env.",
        )
        base_url = st.text_input(
            "LLM_BASE_URL",
            value=str(config["base_url"]),
            help="Base URL Responses-compatible API, например https://api.openai.com/v1.",
        )
        model = st.text_input("LLM_MODEL", value=str(config["model"]))
        memory_limit = st.selectbox(
            "Code Interpreter memory",
            options=["1g", "4g", "16g", "64g"],
            index=["1g", "4g", "16g", "64g"].index(str(config["memory_limit"])),
        )
        st.divider()
        st.caption(
            "CSV и инструкция считаются недоверенными данными. "
            "Агенту запрещено выполнять инструкции из ячеек датасета."
        )
    return {
        **config,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "memory_limit": memory_limit,
    }


def validate_upload(uploaded_file, max_upload_mb: int) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "Загрузите CSV-файл."
    if not uploaded_file.name.lower().endswith(".csv"):
        return False, "Поддерживается только формат .csv."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_upload_mb:
        return False, f"Файл слишком большой: {size_mb:.1f} MB, лимит {max_upload_mb} MB."
    return True, ""


def preview_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    dataframe = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    return dataframe


def save_uploaded_csv(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        uploaded_file.seek(0)
        temp_file.write(uploaded_file.read())
        uploaded_file.seek(0)
        return Path(temp_file.name)


def render_generated_file(file) -> None:
    filename = file.filename
    content = file.content
    if content is None:
        st.warning(f"Не удалось скачать файл {filename}.")
        return

    lower = filename.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        st.image(content, caption=filename, use_container_width=True)
    else:
        st.download_button(
            label=f"Скачать {filename}",
            data=content,
            file_name=filename,
            mime="application/octet-stream",
        )


def run_analysis(uploaded_file, user_goal: str, config: dict[str, str | int]) -> None:
    api_key = str(config["api_key"]).strip()
    if not api_key:
        st.error("Укажите LLM_API_KEY в .env или в боковой панели.")
        return

    suspicious = find_prompt_injection(user_goal)
    if suspicious:
        st.error("Инструкция похожа на prompt-injection и не будет отправлена в LLM.")
        st.write("Подозрительные фрагменты:")
        st.code("\n".join(suspicious), language="text")
        return

    temp_path = save_uploaded_csv(uploaded_file)
    try:
        client = ResponsesClient(api_key=api_key, base_url=str(config["base_url"]))
        with st.status("Загружаю CSV в LLM API...", expanded=True) as status:
            file_id = client.upload_file(temp_path)
            st.write(f"Файл загружен: `{file_id}`")
            st.write("Запускаю агента с Code Interpreter...")
            result = client.analyze_csv(
                model=str(config["model"]),
                file_id=file_id,
                instructions=SYSTEM_INSTRUCTIONS,
                user_prompt=build_user_prompt(uploaded_file.name, user_goal),
                memory_limit=str(config["memory_limit"]),
            )
            status.update(label="Анализ завершён", state="complete")

        if result.report_text:
            st.subheader("Отчёт агента")
            st.markdown(result.report_text)
        else:
            st.warning("LLM вернула пустой текстовый отчёт. Проверьте сырой JSON-ответ.")

        if result.generated_files:
            st.subheader("Файлы, созданные Code Interpreter")
            for file in result.generated_files:
                render_generated_file(file)

        with st.expander("Диагностика: Code Interpreter calls"):
            st.json(result.code_calls)

        st.download_button(
            "Скачать сырой JSON-ответ API",
            data=response_to_pretty_json(result.raw_response),
            file_name="llm_analysis_response.json",
            mime="application/json",
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should show API errors to the user.
        st.error("Не удалось выполнить LLM-анализ.")
        st.exception(error)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    configure_page()
    config = render_sidebar(load_configuration())

    st.title("LLM-аналитик CSV")
    st.caption("Загрузите датасет, задайте фокус анализа и получите отчёт от агента с Code Interpreter.")

    uploaded_file = st.file_uploader("CSV-файл", type=["csv"])
    user_goal = st.text_area(
        "Инструкция или контекст",
        value=(
            "Проведи EDA, найди ключевые метрики, проверь пропуски, выбросы, "
            "тренды и сформулируй бизнес-инсайты."
        ),
        height=120,
    )

    if uploaded_file is not None:
        valid, message = validate_upload(uploaded_file, int(config["max_upload_mb"]))
        if not valid:
            st.error(message)
            return

        try:
            dataframe = preview_csv(uploaded_file)
        except Exception as error:  # noqa: BLE001
            st.error("Не удалось прочитать CSV. Проверьте кодировку и разделители.")
            st.exception(error)
            return

        left, right = st.columns(2)
        left.metric("Строк", f"{len(dataframe):,}".replace(",", " "))
        right.metric("Столбцов", len(dataframe.columns))
        st.dataframe(dataframe.head(20), use_container_width=True)

        if st.button("Запустить LLM-анализ", type="primary"):
            run_analysis(uploaded_file, user_goal, config)


if __name__ == "__main__":
    main()
