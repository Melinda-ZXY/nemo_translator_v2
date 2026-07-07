import streamlit as st

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
]


st.set_page_config(page_title="Nemo V2 翻译器", layout="centered")

st.title("Nemo V2 翻译器")
st.caption("使用第二版 Nemo 词表进行短中文句子的规则翻译。")

example = st.selectbox("示例", EXAMPLES, index=0)
text = st.text_input("中文输入", value=example, placeholder="输入一句短中文，例如：尼莫想和主人玩")

result = translate(text)

st.subheader("发音输出")
st.code(result["nemo"] or " ", language=None)

st.subheader("文字输出")
st.markdown(render_orthography_html(result["nemo"]), unsafe_allow_html=True)

with st.expander("词条解析", expanded=False):
    token_rows = [
        {
            "中文": token.get("text", ""),
            "尼莫词": token.get("nemo", ""),
            "词性": token.get("pos", ""),
            "英文": token.get("english", ""),
            "结构": token.get("shape", ""),
            "已定义": token.get("known", False),
        }
        for token in result["tokens"]
    ]
    st.dataframe(token_rows, hide_index=True, use_container_width=True)

with st.expander("V2 词表", expanded=False):
    st.dataframe(lexicon_rows(), hide_index=True, use_container_width=True)

with st.expander("Raw JSON", expanded=False):
    st.code(to_json(result), language="json")

st.info("文字输出区会保留待补字形占位符；等 v2 glyph map 完成后可以直接替换。")
