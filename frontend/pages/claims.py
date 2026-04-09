import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api

st.title("Claims")
st.markdown("---")

# ── Fetch transactions for the dropdown ───────────────────────────────────────
with st.spinner("Loading data..."):
    tx_status, tx_data = api.list_transactions(st.session_state.token)
    cl_status, cl_data = api.list_claims(st.session_state.token)

transactions = tx_data if tx_status == 200 and isinstance(tx_data, list) else []
claims = cl_data if cl_status == 200 and isinstance(cl_data, list) else []

# ── File a New Claim ──────────────────────────────────────────────────────────
st.subheader("File a Dispute Claim")

if not transactions:
    st.info("You have no transactions to dispute.")
else:
    tx_options = {
        f"TX #{t['id']} — ${t['amount']:.2f} @ {t['location']} [{t['risk_level']}]": t
        for t in transactions
    }

    with st.form("claim_form"):
        selected_label = st.selectbox("Select Transaction to Dispute", list(tx_options.keys()))
        reason = st.text_area("Reason for Dispute", placeholder="Describe why you're disputing this transaction...")
        claim_amount = st.number_input(
            "Claim Amount ($)",
            min_value=0.01,
            max_value=float(tx_options[selected_label]["amount"]),
            value=float(tx_options[selected_label]["amount"]),
            step=0.01,
            format="%.2f",
        )
        submit_claim = st.form_submit_button("Submit Claim", use_container_width=True)

    if submit_claim:
        if not reason.strip():
            st.error("Please provide a reason for the dispute.")
        else:
            selected_tx = tx_options[selected_label]
            with st.spinner("Submitting claim..."):
                status, data = api.create_claim(
                    st.session_state.token,
                    selected_tx["id"],
                    reason,
                    claim_amount,
                )
            if status == 201:
                result_status = data["status"]
                if result_status == "APPROVED":
                    st.success(f"Claim APPROVED — your dispute has been accepted.")
                elif result_status == "MANUAL_REVIEW":
                    st.warning(f"Claim requires MANUAL REVIEW — a staff member will investigate.")
                else:
                    st.error(f"Claim REJECTED — does not meet dispute criteria.")
            else:
                st.error(data.get("detail", "Failed to submit claim."))

st.markdown("---")

# ── Claims History ────────────────────────────────────────────────────────────
st.subheader("Your Claims")

if not claims:
    st.info("No claims filed yet.")
    st.stop()

df = pd.DataFrame(claims)
df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
df["amount"] = df["amount"].round(2)

status_filter = st.selectbox("Filter by Status", ["All", "PENDING", "APPROVED", "REJECTED", "MANUAL_REVIEW"])
filtered = df if status_filter == "All" else df[df["status"] == status_filter]

st.caption(f"Showing {len(filtered)} of {len(df)} claims")

STATUS_COLORS = {
    "APPROVED": "color: green; font-weight: bold",
    "REJECTED": "color: red; font-weight: bold",
    "MANUAL_REVIEW": "color: orange; font-weight: bold",
    "PENDING": "color: gray",
}

def color_status(val):
    return STATUS_COLORS.get(val, "")

styled = filtered.style.applymap(color_status, subset=["status"])
st.dataframe(styled, use_container_width=True, hide_index=True)
