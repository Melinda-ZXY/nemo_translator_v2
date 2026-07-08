from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path


GLYPH_DIR = Path(__file__).parent / "assets" / "version2"


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
  gap: 12px;
  align-items: flex-end;
  margin: 4px 0 12px;
}
.v2-glyph {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  min-width: 58px;
}
.v2-glyph img {
  display: block;
  max-width: 92px;
  height: 78px;
  object-fit: contain;
}
.v2-glyph-label {
  margin-top: 4px;
  color: #475569;
  font-size: 11px;
  line-height: 1;
}
.v2-glyph-placeholder {
  min-width: 74px;
  height: 78px;
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
        parts.append(_render_token(token))

    parts.append("</div>")
    return "".join(parts)


def _render_token(token: str) -> str:
    safe_token = html.escape(token)
    data_uri = _glyph_data_uri(token)
    if data_uri:
        return (
            f'<span class="v2-glyph" title="{safe_token}">'
            f'<img src="{data_uri}" alt="{safe_token}"/>'
            f'<span class="v2-glyph-label">{safe_token}</span>'
            "</span>"
        )

    return (
        f'<span class="v2-glyph-placeholder" title="待补字形：{safe_token}">'
        f'<span class="v2-glyph-token">{safe_token}</span>'
        '<span class="v2-glyph-note">待补字形</span>'
        "</span>"
    )


@lru_cache(maxsize=None)
def _glyph_data_uri(token: str) -> str | None:
    path = GLYPH_DIR / f"{token.lower()}.svg"
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
