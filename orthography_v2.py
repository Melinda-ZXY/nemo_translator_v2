from __future__ import annotations

import html


GLYPH_MAP: dict[str, str] = {}


def render_orthography_html(nemo_text: str) -> str:
    tokens = [token for token in (nemo_text or "").split() if token]
    if not tokens:
        return '<div class="v2-orthography-empty">暂无输出</div>'

    parts = [
        """
<style>
.v2-orthography-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin: 4px 0 12px;
}
.v2-glyph-placeholder {
  min-width: 74px;
  height: 58px;
  padding: 7px 9px;
  border: 1px dashed #9aa4b2;
  border-radius: 8px;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #f8fafc;
  color: #334155;
  line-height: 1.1;
}
.v2-glyph-token {
  font-size: 16px;
  font-weight: 650;
}
.v2-glyph-note {
  font-size: 11px;
  color: #64748b;
  margin-top: 5px;
}
.v2-orthography-empty {
  color: #64748b;
}
</style>
<div class="v2-orthography-row">
"""
    ]

    for token in tokens:
        normalized = token.lower()
        glyph = GLYPH_MAP.get(normalized)
        if glyph:
            parts.append(glyph)
            continue
        safe_token = html.escape(token)
        parts.append(
            f"""
  <span class="v2-glyph-placeholder" title="待补字形：{safe_token}">
    <span class="v2-glyph-token">{safe_token}</span>
    <span class="v2-glyph-note">待补字形</span>
  </span>
"""
        )

    parts.append("</div>")
    return "".join(parts)
