import streamlit as st

from audio_postprocess import process_wav_bytes
from fish_tts import nemo_to_fish_tts_text, synthesize_fish_tts
from orthography_v2 import render_orthography_html
from translator_v2_core import lexicon_rows, to_json, translate


VOICE_MODE_LABELS = {
    "normal": "normal",
    "angry": "angry",
}

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
    "搞什么鬼！",
    "那里",
    "这里",
    "我很得意",
    "电池上升",
    "电池下降",
    "这里没事的",
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
tts_default = nemo_to_fish_tts_text(result["nemo"])
if st.session_state.get("last_fish_tts_default") != tts_default:
    st.session_state["fish_tts_text"] = tts_default
    st.session_state["last_fish_tts_default"] = tts_default

tts_text = st.text_input("Fish Audio 输入", key="fish_tts_text")
voice_mode = st.selectbox("语音模式", list(VOICE_MODE_LABELS), format_func=VOICE_MODE_LABELS.get)
angry_speed = st.slider("angry 语速", min_value=1.0, max_value=1.8, value=1.6, step=0.05)
secret_fish_api_key = st.secrets.get("FISH_API_KEY", "")
secret_fish_reference_id = st.secrets.get("FISH_REFERENCE_ID", st.secrets.get("FISH_SPEAKER_ID", ""))
secret_fish_model = st.secrets.get("FISH_MODEL", "s2.1-pro")

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
    fish_model_input = st.text_input("Model", value=secret_fish_model or "s2.1-pro")

fish_api_key = fish_api_key_input.strip() or secret_fish_api_key
fish_reference_id = fish_reference_id_input.strip() or secret_fish_reference_id
fish_model = fish_model_input.strip() or secret_fish_model or "s2.1-pro"

fish_ready = bool(fish_api_key and fish_reference_id)
if not fish_ready:
    st.warning("请在 Fish Audio 设置里填写 API Key 和 Speaker / Reference ID，或在 Streamlit Secrets 里设置。")

if st.button("生成语音", disabled=not bool(tts_text.strip()) or not fish_ready):
    try:
        with st.spinner("正在生成语音..."):
            tts_speed = angry_speed if voice_mode == "angry" else 1.0
            audio = synthesize_fish_tts(
                tts_text,
                api_key=fish_api_key,
                reference_id=fish_reference_id,
                model=fish_model,
                speed=tts_speed,
                audio_format="wav",
            )
            if voice_mode == "angry":
                processed_audio = process_wav_bytes(audio.audio_bytes, syllable_text=tts_text)
                audio_bytes = processed_audio.wav_bytes
                audio_mime_type = processed_audio.mime_type
                caption = (
                    f"模式：angry，语速：{tts_speed:.2f}x，"
                    f"结尾托住：{'yes' if processed_audio.tail_adjusted else 'no'}，"
                    f"最后词：{processed_audio.final_word or 'unknown'}"
                )
            else:
                audio_bytes = audio.audio_bytes
                audio_mime_type = audio.mime_type
                caption = "模式：normal，语速：1.00x"
        st.audio(audio_bytes, format=audio_mime_type)
        st.caption(caption)
    except Exception as exc:
        st.error(f"Fish Audio 生成失败：{exc}")

with st.expander("V2 词表", expanded=False):
    st.dataframe(lexicon_rows(), hide_index=True, use_container_width=True)

with st.expander("Raw JSON", expanded=False):
    st.code(to_json(result), language="json")

st.info("文字输出会使用 Version2 字形；没有对应 SVG 的词会保留待补字形占位符。")
