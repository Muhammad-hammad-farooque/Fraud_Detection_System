import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api

st.title("Fraud Detection System")
st.markdown("---")

login_tab, register_tab = st.tabs(["Login", "Register"])

# ── Login ─────────────────────────────────────────────────────────────────────
with login_tab:
    st.subheader("Sign in to your account")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields.")
        else:
            with st.spinner("Signing in..."):
                status, data = api.login(email, password)
            if status == 200:
                st.session_state.token = data["access_token"]
                _, user_data = api.get_me(st.session_state.token)
                st.session_state.user = user_data
                st.success(f"Welcome back, {user_data['name']}!")
                st.rerun()
            else:
                st.error(data.get("detail", "Login failed."))

# ── Register ──────────────────────────────────────────────────────────────────
with register_tab:
    st.subheader("Create a new account")
    with st.form("register_form"):
        name = st.text_input("Full Name")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        reg_submitted = st.form_submit_button("Register", use_container_width=True)

    if reg_submitted:
        if not name or not reg_email or not reg_password:
            st.error("Please fill in all fields.")
        elif reg_password != reg_confirm:
            st.error("Passwords do not match.")
        else:
            with st.spinner("Creating account..."):
                status, data = api.register(name, reg_email, reg_password)
            if status == 201:
                st.success("Account created! Please log in.")
            else:
                st.error(data.get("detail", "Registration failed."))
