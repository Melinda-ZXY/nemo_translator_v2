import streamlit as st

from fish_tts import synthesize_fish_tts
from orthography_v2 import render_orthography_html
from translator_v2_core import lexicon_rows, to_json, translate


EXAMPLES = [
    "主人开心",
    "主人不开心",
    "尼莫想和主人玩",
    "主人想吃食物",
    "主人喜欢尼莫",
    "尼莫看到电池",
    "我的电池",
    "主人快离开",
    "主人紧张吗",
    "主人是鬼",
]


st.set_page_config(page_title="Nemo V2 翻译器", layout="centered")

st.title("Nemo V2 翻译器")

example = st.selectbox("示例", EXAMPLES, index=0)
text = st.text_input("中文输入", value=example, placeholder="输入一句短中文，例如：尼莫想和主人玩")

result = translate(text)

st.subheader("发音输出")
st.code(result["nemo"] or " ", language=None)

st.subheader("文字输出")
st.markdown(render_orthography_html(result["nemo"]), unsafe_allow_html=True)

st.subheader("语音输出")
tts_default = result["nemo"]
if st.session_state.get("last_fish_tts_default") != tts_default:
    st.session_state["fish_tts_text"] = tts_default
    st.session_state["last_fish_tts_default"] = tts_default

tts_text = st.text_input("Fish Audio 输入", key="fish_tts_text")
tts_speed = st.slider("语速", min_value=0.5, max_value=1.8, value=1.0, step=0.05)
secret_fish_api_key = st.secrets.get("FISH_API_KEY", "")
secret_fish_reference_id = st.secrets.get("FISH_REFERENCE_ID", st.secrets.get("FISH_SPEAKER_ID", ""))
secret_fish_model = st.secrets.get("FISH_MODEL", "s2-pro")

with st.expander("Fish Audio 设置", expanded=not bool(secret_fish_api_key and secret_fish_reference_id)):
    fish_api_key_input = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="如果没有设置 Streamlit Secrets，可以临时填在这里",
    )
    fish_reference_id_input = st.text_input(
        "Speaker / Reference ID",
        value="",
        placeholder="如果没有设置 Streamlit Secrets，可以临时填在这里",
    )
    fish_model_input = st.text_input("Model", value=secret_fish_model or "s2-pro")

fish_api_key = fish_api_key_input.strip() or secret_fish_api_key
fish_reference_id = fish_reference_id_input.strip() or secret_fish_reference_id
fish_model = fish_model_input.strip() or secret_fish_model or "s2-pro"

fish_ready = bool(fish_api_key and fish_reference_id)
if not fish_ready:
    st.warning("请在 Fish Audio 设置里填写 API Key 和 Speaker / Reference ID，或在 Streamlit Secrets 里设置。")

if st.button("生成语音", disabled=not bool(tts_text.strip()) or not fish_ready):
    try:
        with st.spinner("正在生成语音..."):
            audio = synthesize_fish_tts(
                tts_text,
                api_key=fish_api_key,
                reference_id=fish_reference_id,
                model=fish_model,
                speed=tts_speed,
            )
        st.audio(audio.audio_bytes, format=audio.mime_type)
    except Exception as exc:
        st.error(f"Fish Audio 生成失败：{exc}")

with st.expander("V2 词表", expanded=False):
    st.dataframe(lexicon_rows(), hide_index=True, use_container_width=True)

with st.expander("Raw JSON", expanded=False):
    st.code(to_json(result), language="json")

st.info("文字输出会使用 Version2 字形；没有对应 SVG 的词会保留待补字形占位符。")
