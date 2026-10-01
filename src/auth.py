from __future__ import annotations

import hmac
import streamlit as st

from .config import APP_PASSWORD


def require_password() -> bool:
    if not APP_PASSWORD:
        st.error(
            "APP_PASSWORD is not configured. Set it as a deployment secret before using AgriRAG."
        )
        return False

    if st.session_state.get("authenticated"):
        return True

    st.title("AgriRAG")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False
