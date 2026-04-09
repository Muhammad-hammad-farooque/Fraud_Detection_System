import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api

st.title("Dashboard")
st.markdown(f"Welcome back, **{st.session_state.user['name']}**")
st.markdown("---")

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    tx_status, tx_data = api.list_transactions(st.session_state.token)
    cl_status, cl_data = api.list_claims(st.session_state.token)

if tx_status != 200:
    st.error("Failed to load transactions.")
    st.stop()

transactions = tx_data if isinstance(tx_data, list) else []
claims = cl_data if isinstance(cl_data, list) else []

# ── Key Metrics ───────────────────────────────────────────────────────────────
total = len(transactions)
fraud_count = sum(1 for t in transactions if t["is_fraud"])
pending_claims = sum(1 for c in claims if c["status"] == "PENDING")
high_risk = sum(1 for t in transactions if t["risk_level"] == "HIGH")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total)
col2.metric("Flagged as Fraud", fraud_count, delta=f"{round(fraud_count/total*100, 1)}% of total" if total else None, delta_color="inverse")
col3.metric("High Risk", high_risk)
col4.metric("Pending Claims", pending_claims)

st.markdown("---")

if not transactions:
    st.info("No transactions yet. Submit your first transaction in the Transactions page.")
    st.stop()

df = pd.DataFrame(transactions)
df["created_at"] = pd.to_datetime(df["created_at"])

# ── Risk Level Distribution ───────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risk_counts["order"] = risk_counts["Risk Level"].map(risk_order)
    risk_counts = risk_counts.sort_values("order").drop("order", axis=1)
    st.bar_chart(risk_counts.set_index("Risk Level"))

with col_right:
    st.subheader("Decision Breakdown")
    decision_counts = df["decision"].value_counts().reset_index()
    decision_counts.columns = ["Decision", "Count"]
    st.bar_chart(decision_counts.set_index("Decision"))

# ── Transaction Volume Over Time ──────────────────────────────────────────────
st.subheader("Transaction Amount Over Time")
df_sorted = df.sort_values("created_at")
df_sorted["date"] = df_sorted["created_at"].dt.date
daily = df_sorted.groupby("date")["amount"].sum().reset_index()
st.line_chart(daily.set_index("date"))

# ── Recent Transactions ───────────────────────────────────────────────────────
st.subheader("Recent Transactions")

def style_row(row):
    if row["is_fraud"]:
        return ["background-color: #ffcccc"] * len(row)
    elif row["risk_level"] == "MEDIUM":
        return ["background-color: #fff3cc"] * len(row)
    return [""] * len(row)

display_cols = ["id", "amount", "location", "device_id", "risk_score", "risk_level", "decision", "is_fraud", "created_at"]
recent = df.sort_values("created_at", ascending=False).head(10)[display_cols].copy()
recent["risk_score"] = recent["risk_score"].round(3)
recent["created_at"] = recent["created_at"].dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(
    recent.style.apply(style_row, axis=1),
    use_container_width=True,
    hide_index=True,
)
