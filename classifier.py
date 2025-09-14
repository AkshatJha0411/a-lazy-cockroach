# classifier.py
import os
import groq
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use a simple dictionary for in-memory caching
classification_cache = {}

# More complex examples for reference
CLASSIFICATION_EXAMPLES = {
    "My new connector is failing to authenticate with my data source. This is a major issue.": {
        "topic": "Connector",
        "sentiment": "Negative",
        "priority": "High"
    },
    "I'm looking for documentation on Atlan's security features, specifically around SSO.": {
        "topic": "Security",
        "sentiment": "Neutral",
        "priority": "Normal"
    },
    "The new product dashboard is fantastic and really intuitive to use!": {
        "topic": "Product",
        "sentiment": "Positive",
        "priority": "Normal"
    },
    "I've been trying to follow the steps in the guide to set up a new governance workflow, but I'm getting stuck.": {
        "topic": "How-to",
        "sentiment": "Negative",
        "priority": "Normal"
    },
    "Can you help me understand how Atlan handles invoicing for our enterprise plan?": {
        "topic": "Billing",
        "sentiment": "Neutral",
        "priority": "Normal"
    },
    "I need to implement a solution for data quality checks using the Atlan SDK. Are there any examples?": {
        "topic": "API/SDK",
        "sentiment": "Neutral",
        "priority": "Normal"
    }
}

def classify_ticket(query: str):
    """
    Uses an LLM to classify a user's query into a structured format.
    """
    if query in classification_cache:
        return classification_cache[query]

    # Dynamically build the prompt with all examples
    example_prompts = ""
    for example_query, example_output in CLASSIFICATION_EXAMPLES.items():
        example_prompts += f"""
    Example Query: "{example_query}"
    Example Output: {json.dumps(example_output)}
    """

    prompt = f"""
    You are a highly skilled support ticket classifier. Your task is to analyze a user's query and classify it into a JSON object with the following keys:
    - topic: (choose from "How-to", "Product", "Best practices", "API/SDK", "SSO", "Connector", "Billing", "Security", "Lineage", "Glossary" or "Other")
    - sentiment: (choose from "Positive", "Neutral", "Negative")
    - priority: (choose from "High", "Normal", "Low")
    {example_prompts}
    Query to classify: "{query}"

    Provide only the JSON object in your response.
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    try:
        classification = json.loads(completion.choices[0].message.content)
        classification_cache[query] = classification
        return classification
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}")
        # Fallback to a default classification
        return {
            "topic": "Other",
            "sentiment": "Neutral",
            "priority": "Normal"
        }
