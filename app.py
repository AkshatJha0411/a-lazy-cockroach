# app.py
import streamlit as st
from classifier import classify_ticket
from rag_pipeline import query_rag

st.set_page_config(layout="wide", page_title="Support AI Copilot")

st.title("📩 Support AI Copilot")

# Create a clear button at the top-right
def clear_form():
    st.session_state.query = ""

col_title, col_button = st.columns([0.8, 0.2])
with col_title:
    query = st.text_input("Enter your ticket/query:", key="query")
with col_button:
    st.button("Clear", on_click=clear_form)

if query:
    # Use a spinner for the classification step
    with st.spinner("Classifying your ticket..."):
        classification = classify_ticket(query)

    # Layout: two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Internal Analysis")
        st.json(classification)

    with col2:
        st.subheader("💬 Final Response")

        if classification["topic"] in ["How-to", "Product", "Best practices", "API/SDK", "SSO", "Connector", "Billing", "Security"]:
            with st.spinner("Finding the best response..."):
                answer, sources = query_rag(query)
                if sources:
                    st.write(answer)
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.markdown(f"- {s}")
                else:
                    st.info(
                        f"This ticket has been classified as a '{classification['topic']}' issue and routed to the appropriate team."
                    )
        else:
            st.info(
                f"This ticket has been classified as a '{classification['topic']}' issue and routed to the appropriate team."
            )
