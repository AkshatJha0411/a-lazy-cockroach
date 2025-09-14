# 📩 Support AI Copilot

This project is a Retrieval-Augmented Generation (RAG) system designed to act as an internal support copilot, providing accurate and contextual answers to user queries based on a corpus of internal documentation.

### Deployed Link : https://customer-support-ticket-handler.streamlit.app

### Architecture Diagram:
<img width="541" height="768" alt="Screenshot 2025-09-14 at 10 18 41 PM" src="https://github.com/user-attachments/assets/e0b620c2-2127-434f-8ab5-c65870f09bd1" />

## 💡 Major Design Decisions and Trade-offs

The core of this application is an AI pipeline that seamlessly integrates several components to deliver a robust solution. The key design decisions and trade-offs made are:

* **RAG Architecture over Fine-Tuning:** Instead of fine-tuning a large language model on our private data, we chose a RAG approach. This is a more cost-effective and agile strategy. It allows us to keep the LLM static and update the knowledge base independently, ensuring the system's responses are always grounded in the most current documentation without the expensive and time-consuming process of retraining.

* **Component Selection:**

  * **LLM (Groq's `llama-3.1-8b-instant`)**: Selected for its exceptional speed and low latency. The instant model is highly suitable for real-time applications like a support copilot, where a quick response is paramount. The trade-off is a potentially smaller context window compared to larger models, but this is mitigated by the efficiency of the RAG retrieval process.

  * **Embedding Model (`Sentence-Transformers/all-MiniLM-L6-v2`)**: Chosen for its balance of performance and efficiency. It provides high-quality semantic embeddings at a much faster rate and with a smaller memory footprint than larger models. While a more complex model might offer marginal gains in specific niche contexts, the performance of this model is more than sufficient for a general-purpose technical documentation corpus.

  * **Vector Database (Pinecone)**: Used for its fully managed, serverless architecture. This choice minimizes infrastructure management overhead and allows the system to scale efficiently with demand without manual intervention.

* **Data Ingestion Pipeline**: The ingestion pipeline is designed for robustness and idempotency. The use of a simple `if not pc.list_indexes()` check prevents redundant index creation, and the caching mechanism in the original script efficiently handles large data volumes while mitigating API rate-limiting issues.

* **Prompting Strategy**: The system uses a simple but effective prompting strategy. It explicitly instructs the LLM to use the provided context and to state "I don't know" if the information is not present. This minimizes the risk of hallucinations, which is a critical trade-off for a support system where factual accuracy is more important than a verbose but incorrect answer.

## 🏛️ Architecture

The system follows a classic RAG architecture, divided into two main phases: an offline data ingestion pipeline and a real-time query pipeline.

### Ingestion Pipeline

The ingestion pipeline is a batch process that runs whenever the source data changes.

1. **Crawl**: The script crawls the `docs.atlan.com` and `developer.atlan.com` sitemaps to collect all relevant text content.

2. **Chunk**: A `RecursiveCharacterTextSplitter` breaks the raw text into smaller, manageable chunks.

3. **Embed**: The `Sentence-Transformers/all-MiniLM-L6-v2` model converts these text chunks into dense vector embeddings.

4. **Upsert**: The vectors and their associated metadata are uploaded to the Pinecone index.

### Query Pipeline

This is the real-time process that handles user queries.

1. **Classify**: An initial rule-based classifier categorizes the user's query to determine if it should go through the RAG pipeline.

2. **Embed Query**: The user's query is converted into a vector embedding using the same model as the ingestion pipeline.

3. **Retrieve**: The query vector is used to search the Pinecone index for the most semantically similar chunks.

4. **Augment & Generate**: The retrieved chunks are added to a prompt as context, which is then sent to the Groq LLM to generate a final, grounded answer.

## 🛠️ Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8 or higher
* A Pinecone API key
* A Groq API key
* A Pinecone index named `atlan-docs` with a dimension of `384` and metric `cosine`. (steps for this in ingest.py)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AkshatJha0411/a-lazy-cockroach
cd <repository-name>
```

### Step 2: Set up the Python Environment

Create and activate a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages.

```bash
pip install streamlit groq pinecone python-dotenv sentence-transformers numpy
```

> Note: The `requirements.txt` file should contain all the necessary libraries, such as `pinecone`, `sentence-transformers`, `groq`, `numpy`, `python-dotenv`, and `streamlit`.

### Step 3: Configure API Keys

Create a `.env` file in the root directory of your project and add your API keys.

```bash
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="your-pinecone-environment"
GROQ_API_KEY="your-groq-api-key"
```

### Step 4: Run the Application

Start the Streamlit application from your terminal.

```bash
streamlit run app.py
```

This will open the **Support AI Copilot** in your browser. You can now enter queries and see the RAG pipeline in action.
