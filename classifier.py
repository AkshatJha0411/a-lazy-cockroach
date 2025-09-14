# classifier.py
def classify_ticket(query: str):
    """
    Rule-based classification for tickets.
    """

    query_lower = query.lower()
    topic = "Other"

    if any(word in query_lower for word in ["how", "steps", "guide"]):
        topic = "How-to"
    elif any(word in query_lower for word in ["product", "feature", "atlan"]):
        topic = "Product"
    elif "best practice" in query_lower:
        topic = "Best practices"
    elif "api" in query_lower or "sdk" in query_lower:
        topic = "API/SDK"
    elif "sso" in query_lower or "single sign on" in query_lower:
        topic = "SSO"
    elif "connector" in query_lower:
        topic = "Connector"
    elif "billing" in query_lower or "invoice" in query_lower:
        topic = "Billing"
    elif "security" in query_lower:
        topic = "Security"

    # Sentiment (naive)
    sentiment = "Negative" if any(w in query_lower for w in ["error", "issue", "fail"]) else "Neutral"

    # Priority (naive)
    priority = "High" if "urgent" in query_lower else "Normal"

    return {
        "topic": topic,
        "sentiment": sentiment,
        "priority": priority
    }
