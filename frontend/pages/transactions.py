import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api

st.title("Transactions")
st.markdown("---")

# ── Submit New Transaction ────────────────────────────────────────────────────
st.subheader("Submit New Transaction")

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, step=0.01, format="%.2f")
    with col2:
        location = st.text_input("Location", placeholder="e.g. New York, USA")
    with col3:
        device_id = st.text_input("Device ID", placeholder="e.g. device-abc-123")
    submitted = st.form_submit_button("Submit Transaction", use_container_width=True)

if submitted:
    if not location or not device_id:
        st.error("Please fill in all fields.")
    else:
        with st.spinner("Processing transaction..."):
            status, data = api.create_transaction(
                st.session_state.token, location, amount, device_id
            )
        if status == 200:
            risk = data["risk_level"]
            decision = data["decision"]
            score = round(data["risk_score"], 3)

            if risk == "HIGH":
                st.error(f"Transaction REJECTED — High risk (score: {score}). Flagged as fraud.")
            elif risk == "MEDIUM":
                st.warning(f"Transaction requires MANUAL CHECK — Medium risk (score: {score}).")
            else:
                st.success(f"Transaction APPROVED — Low risk (score: {score}).")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Transaction ID", f"#{data['id']}")
            col_b.metric("Risk Score", score)
            col_c.metric("Decision", decision)
        else:
            st.error(data.get("detail", "Transaction failed."))

st.markdown("---")

# ── Transaction History ───────────────────────────────────────────────────────
st.subheader("Transaction History")

with st.spinner("Loading transactions..."):
    status, tx_data = api.list_transactions(st.session_state.token)

if status != 200:
    st.error("Failed to load transactions.")
    st.stop()

transactions = tx_data if isinstance(tx_data, list) else []

if not transactions:
    st.info("No transactions yet.")
    st.stop()

df = pd.DataFrame(transactions)
df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
df["risk_score"] = df["risk_score"].round(3)

# Filter controls
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    risk_filter = st.selectbox("Filter by Risk Level", ["All", "LOW", "MEDIUM", "HIGH"])
with col_f2:
    decision_filter = st.selectbox("Filter by Decision", ["All", "ALLOW", "MANUAL_CHECK", "REJECT"])
with col_f3:
    fraud_filter = st.selectbox("Show", ["All", "Fraud only", "Clean only"])

filtered = df.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if decision_filter != "All":
    filtered = filtered[filtered["decision"] == decision_filter]
if fraud_filter == "Fraud only":
    filtered = filtered[filtered["is_fraud"] == True]
elif fraud_filter == "Clean only":
    filtered = filtered[filtered["is_fraud"] == False]

st.caption(f"Showing {len(filtered)} of {len(df)} transactions")

def color_risk(val):
    colors = {"HIGH": "color: red; font-weight: bold", "MEDIUM": "color: orange; font-weight: bold", "LOW": "color: green"}
    return colors.get(val, "")

def color_decision(val):
    colors = {"REJECT": "color: red", "MANUAL_CHECK": "color: orange", "ALLOW": "color: green"}
    return colors.get(val, "")

display_cols = ["id", "amount", "location", "device_id", "risk_score", "risk_level", "decision", "is_fraud", "created_at"]
styled = (
    filtered[display_cols]
    .style
    .map(color_risk, subset=["risk_level"])
    .map(color_decision, subset=["decision"])
)

st.dataframe(styled, use_container_width=True, hide_index=True)
