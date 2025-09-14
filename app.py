# app.py
import streamlit as st
from classifier import classify_ticket
from rag_pipeline import query_rag

st.set_page_config(layout="wide", page_title="Support AI Copilot")

st.title("📩 Support AI Copilot")

query = st.text_input("Enter your ticket/query:")

if query:
    classification = classify_ticket(query)

    # Layout: two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Internal Analysis")
        st.json(classification)

    with col2:
        st.subheader("💬 Final Response")

        if classification["topic"] in ["How-to", "Product", "Best practices", "API/SDK", "SSO"]:
            answer, sources = query_rag(query)
            st.write(answer)
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- {s}")
        else:
            st.info(
                f"This ticket has been classified as a '{classification['topic']}' issue and routed to the appropriate team."
            )
