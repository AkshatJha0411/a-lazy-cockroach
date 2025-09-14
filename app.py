# app.py
import streamlit as st
import json
from classifier import classify_ticket
from rag_pipeline import query_rag

st.set_page_config(layout="wide", page_title="Support AI Copilot")

st.title("📩 Support AI Copilot")

# --- Tabs for navigation ---
tab1, tab2 = st.tabs(["Bulk Classification Dashboard", "Interactive AI Agent"])

with tab1:
    st.header("Bulk Ticket Classification Dashboard")
    # Load pre-classified tickets from a JSON file
    try:
        with open("sample_tickets_output.json", 'r') as f:
            classified_tickets = json.load(f)
    except FileNotFoundError:
        st.error("`sample_tickets_output.json` not found. Please ensure the file is in the same directory.")
        classified_tickets = []
    
    # the below was function for getting output on the bulk tickets (got rate limit error and thus now output is saved in sample_tickets_output.json file)
    # def load_and_classify_tickets(tickets):
    #     for ticket in tickets:
    #         query = f"{ticket['subject']} {ticket['body']}"
    #         ticket['classification'] = classify_ticket(query)
    #     return tickets
    
    for ticket in classified_tickets:
        st.markdown(f"**Ticket ID:** {ticket['id']}")
        st.markdown(f"**Subject:** {ticket['subject']}")
        st.markdown(f"**Query:** {ticket['body']}")

        classification = {
            "topic": ticket["Topic"],
            "sentiment": ticket["Sentiment"],
            "priority": ticket["Priority"]
        }

        col_topic, col_sentiment, col_priority = st.columns(3)
        with col_topic:
            st.metric("Topic", classification["topic"])
        with col_sentiment:
            st.metric("Sentiment", classification["sentiment"])
        with col_priority:
            st.metric("Priority", classification["priority"])
        st.markdown("---")

with tab2:
    st.header("Interactive AI Agent")

    # --- Live Query Input ---
    def clear_form():
        st.session_state.query = ""

    col_title, col_button = st.columns([0.8, 0.2])
    with col_title:
        query = st.text_input("Enter a new ticket/query:", key="query")
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
            
            rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
            
            if classification["topic"] in rag_topics:
                with st.spinner("Finding the best response..."):
                    answer, sources = query_rag(query)
                    if "I don't know" not in answer:
                        st.write(answer)
                        st.markdown("**Sources:**")
                        for s in sources:
                            st.markdown(f"- {s}")
                    else:
                        st.info(
                            f"This ticket has been classified as a '{classification['topic']}' issue and is currently out of the scope of this AI agent. It has been routed to the appropriate team for assistance."
                        )
            else:
                st.info(
                    f"This ticket has been classified as a '{classification['topic']}' issue and is currently out of the scope of this AI agent. It has been routed to the appropriate team for assistance."
                )
