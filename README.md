# TA LLM: Retrieval-Augmented Generation Chat Application

TA LLM is a chat-based application designed to answer questions about **Introduction to Deep Learning and LLM-based Generative AI Systems** using retrieval-augmented generation (RAG). The application integrates OpenAI's GPT-4o-mini models, ChromaDB for document retrieval, and a Chain-of-Thought (CoT) reasoning mechanism to provide comprehensive and accurate responses.

---

## Features

- **No Retrieval**: Directly generates answers without relying on external context.
- **RAG**: Uses ChromaDB to retrieve contextually relevant documents to augment responses.
- **RAG + CoT**: Breaks down the question step-by-step with a Chain-of-Thought reasoning prompt.
- **RAG + Self-Consistency**: Evaluates multiple generated responses and selects the best one.

---

## Prerequisites

1. Python 3.8 or higher
2. A valid OpenAI API key
3. A `chromadb` instance (installed locally or through pip)
4. Basic familiarity with Flask and Streamlit

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

3. **OpenAI API Key**
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"

## Starting the Server
1. **Run the Flask Backend**
    ```bash
    python server.py

The server will start on http://localhost:5000.

2. **Verify Server Connectivity**
    ```bash
    curl -X POST http://localhost:5000/run-rag \
     -H "Content-Type: application/json" \
     -d '{"query": "What is deep learning?", "model": "gpt-4o-mini", "search_type": "RAG"}'

## Starting the Frontend
    ```bash
    streamlit run ui.py

Visit the Streamlit app in your browser at http://localhost:8501.


## Using the Application
1.	Choose a Model
    -	gpt-4o-mini: Base model
    -	ft:gpt-4o-mini-2024-07-18:f-prime-capital::AbZYSjIT: Fine-tuned model for this specific domain.
2.	Select a Search Type
    -	No Retrieval: Generates answers without additional context.
    -	RAG: Retrieves relevant documents from ChromaDB to improve response accuracy.
    -	RAG + CoT: Incorporates step-by-step reasoning for complex questions.
    -	RAG + Self-Consistency: Aggregates multiple responses to evaluate and return the best.
3.	Submit a Query
    -	Enter your question in the text box and click Send.
    -	View the conversation in the chat window, with your query and the model’s response displayed.
4.	Clear Chat History
    -	Click the Clear Chat button to reset the conversation.