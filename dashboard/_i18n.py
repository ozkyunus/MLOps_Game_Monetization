"""Tiny i18n helper — one shared language toggle across all Streamlit pages.

Usage in any page:
    from _i18n import L, sidebar_lang_toggle
    sidebar_lang_toggle()
    st.title(L("Türkçe başlık", "English title"))
"""
from __future__ import annotations

import streamlit as st


def get_lang() -> str:
    return st.session_state.get("lang", "tr")


def L(tr: str, en: str) -> str:
    """Return the string in the currently-selected language."""
    return tr if get_lang() == "tr" else en


def sidebar_lang_toggle() -> None:
    """Render the language toggle at the top of the sidebar.
    Call this on every page (Streamlit re-runs each page independently)."""
    current = get_lang()
    with st.sidebar:
        picked = st.radio(
            "🌐  Language / Dil",
            options=["tr", "en"],
            format_func=lambda x: "🇹🇷 Türkçe" if x == "tr" else "🇬🇧 English",
            horizontal=True,
            index=0 if current == "tr" else 1,
            key=f"lang_toggle_{st.session_state.get('_page_hash', 'default')}",
        )
        st.session_state["lang"] = picked
        st.divider()
