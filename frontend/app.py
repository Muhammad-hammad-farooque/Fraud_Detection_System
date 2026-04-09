import streamlit as st

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None

# ── Navigation ────────────────────────────────────────────────────────────────
if not st.session_state.token:
    pg = st.navigation(
        [st.Page("pages/login.py", title="Login", icon="🔑")],
        position="hidden",
    )
else:
    with st.sidebar:
        st.markdown(f"**{st.session_state.user['name']}**")
        st.caption(st.session_state.user["email"])
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

    pg = st.navigation([
        st.Page("pages/dashboard.py",    title="Dashboard",    icon="📊"),
        st.Page("pages/transactions.py", title="Transactions", icon="💳"),
        st.Page("pages/claims.py",       title="Claims",       icon="📋"),
    ])

pg.run()
