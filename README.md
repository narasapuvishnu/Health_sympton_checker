# Healthcare Symptom Checker with Medical Knowledge Retrieval

## Problem Statement
Finding reliable medical information online is difficult and often overwhelming. Standard search engines can return conflicting or overly alarming results, while general-purpose Large Language Models (LLMs) can hallucinate medical facts, posing significant risks. 

## Project Objective
This capstone project builds an AI-powered healthcare symptom checker that accepts symptoms described by a user in natural language and retrieves relevant information from a curated, trusted medical knowledge base. By utilizing Retrieval-Augmented Generation (RAG), the LLM generates answers strictly based on the retrieved context rather than relying solely on internal knowledge, minimizing hallucinations.

**Disclaimer:** This system is for educational purposes only. It is not designed to diagnose users, prescribe medications, or replace a doctor.

## Features
- **Semantic Retrieval**: Uses local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) to perform semantic search over trusted medical documents.
- **RAG Pipeline**: Integrates with Groq's high-speed inference API (Llama3) to generate structured, educational responses.
- **Safety First**: Implements a safety layer to detect potential emergency keywords and advise users to seek immediate professional care.
- **Source Citations**: Transparently displays the sources used to generate the response.
- **Modular Architecture**: Clean separation of concerns (Ingestion, Embeddings, Vector Store, Retrieval, LLM generation, UI).
- **Streamlit Interface**: A clean, professional UI for easy interaction.

## System Architecture

```mermaid
flowchart LR
    subgraph Offline Knowledge Ingestion
        MKS[Medical Knowledge Sources\n(PDF, TXT, MD)] --> DL[Document Loader]
        DL --> TP[Text Processing & Cleaning]
        TP --> C[Chunking]
        C --> EM[Embedding Model]
        EM --> QDRANT[(Qdrant Vector DB)]
    end
    
    subgraph Real-Time Application Flow
        USER[User] -->|Symptom Input| UI[Streamlit UI]
        UI --> SP[Symptom Processing & Safety Check]
        SP -->|If Emergency| ER[Emergency Warning]
        ER --> UI
        SP -->|If Safe| QE[Query Embedding]
        QE --> QS[Qdrant Search]
        QDRANT -.->|Retrieve Chunks| QS
        QS --> RC[Retrieved Context]
        RC --> PR[RAG Prompt]
        PR --> GROQ[Groq LLM]
        GROQ --> SR[Structured Response]
        SR --> UI
    end
```

## Technology Stack
- **Frontend**: Streamlit
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) via HuggingFace
- **Vector Database**: Qdrant (Local or Cloud)
- **LLM**: Groq API (`llama3-8b-8192`)
- **Document Processing**: Langchain (Text Splitters), PyPDF
- **Testing**: Pytest

## Project Structure
```text
healthcare-symptom-checker/
│
├── app.py                      # Main Streamlit application
├── config/
│   └── settings.py             # Environment configuration
├── ingestion/                  # Offline knowledge ingestion module
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   └── ingest.py
├── embeddings/                 # Embedding service
│   └── embedding_service.py
├── vectorstore/                # Qdrant client wrapper
│   └── qdrant_client.py
├── retrieval/                  # Semantic search integration
│   └── retriever.py
├── llm/                        # Groq client and prompts
│   ├── groq_client.py
│   └── prompts.py
├── safety/                     # Safety layer for emergency detection
│   └── safety_checker.py
├── pipeline/                   # RAG pipeline orchestration
│   └── rag_pipeline.py
├── models/                     # Pydantic schemas
│   └── schemas.py
├── utils/
│   └── logger.py
├── data/
│   ├── raw/                    # Place medical documents (TXT/MD/PDF) here
│   └── processed/
├── tests/                      # Unit tests
│   ├── test_chunking.py
│   └── test_safety.py
├── scripts/                    # Helper scripts
│   └── ingest_knowledge.py
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
└── README.md
```

## Prerequisites
- Python 3.9 or higher
- A Groq API Key (Available for free at https://console.groq.com)
- (Optional) Qdrant Cloud API Key. If not provided, it falls back to a local storage directory (`qdrant_storage/`).

## Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone <repo_url>
   cd healthcare-symptom-checker
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Create a `.env` file in the root directory based on `.env.example`:
```bash
cp .env.example .env
```
Open `.env` and fill in your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Knowledge-Base Ingestion
Before running the application, you need to populate the vector database with medical knowledge. A sample document is provided in `data/raw/`. 

To ingest knowledge:
```bash
python scripts/ingest_knowledge.py
```
This script will read files from `data/raw/`, split them into chunks, generate local embeddings, and store them in Qdrant.

## Running the Application
To start the Streamlit UI:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

## Running Tests
To run the automated tests:
```bash
pytest tests/
```

## Example Usage
1. Enter a query in the text area: *"I have a fever, cough, and a sore throat for two days."*
2. Click **Check Symptoms**.
3. View the AI-generated educational summary, possible conditions, general info, and source citations retrieved from the knowledge base.

## Limitations
- **Educational Only**: The AI does not have clinical reasoning and cannot ask follow-up questions.
- **Data Dependency**: The quality of the response is strictly bounded by the documents loaded into the `data/raw/` directory.
- **Latency**: Local embedding generation might introduce slight latency depending on CPU capabilities.

## Safety Disclaimer
This information is for educational purposes only and is not a medical diagnosis or a substitute for professional medical advice. Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.

## Future Scope
- Integration with live medical APIs (e.g., PubMed, MedlinePlus).
- Implementation of memory for multi-turn conversational follow-ups.
- Cloud deployment of the Vector Database for scalable production usage.

## Deployment Instructions
The application is ready to be deployed on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository and set `app.py` as the main file path.
4. Go to **Advanced Settings** -> **Secrets** and paste the contents of your `.env` file.
5. Click **Deploy**.

## GitHub Instructions
Do not commit your `.env` file or `qdrant_storage/` directory. The included `.gitignore` handles this.
```bash
git add .
git commit -m "Initial commit: Healthcare Symptom Checker"
git push origin main
```
