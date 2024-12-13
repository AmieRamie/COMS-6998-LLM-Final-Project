from flask import Flask, request, jsonify
import os
import openai
import chromadb
from chromadb.config import Settings
from tiktoken import encoding_for_model

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()

# Initialize ChromaDB client
persistent_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
collection = persistent_client.get_or_create_collection("llm_tutor_collection")

# Define model names
gpt_4o_base = "gpt-4o-mini"
gpt_4o_finetuned = "ft:gpt-4o-mini-2024-07-18:f-prime-capital::AbZYSjIT"

# Function for embedding generation
def get_embedding(text, model="text-embedding-3-large", max_tokens=8192):
    tokenizer = encoding_for_model(model)
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        return None

# Retrieval function
def retrieve_relevant_documents(query, n_results=5):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    retrieved_text = " ".join(doc[0] for doc in results["documents"])
    return retrieved_text

#No Retrieval
def run_rag_no_retrieval(model, query):
    all_messages = [
        {"role": "system", "content": "You are a helpful tutor who answers questions about a class called Introduction to Deep Learning and LLM based Generative AI Systems"},
        {"role": "user", "content": f"Generate an answer to the following question:\n\n {query}"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        max_tokens=1500,
    )
    return response.choices[0].message.content

# RAG methods
def run_rag(model, query):
    retrieved_text = retrieve_relevant_documents(query)
    all_messages = [
        {"role": "system", "content": "You are a helpful tutor who answers questions about a class called Introduction to Deep Learning and LLM based Generative AI Systems"},
        {"role": "user", "content": f"Generate an answer to the following question using the given context:\n\n {query}\n\n {"="*50}\n\nCONTEXT: {retrieved_text}\n\n"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        max_tokens=1500,
    )
    return response.choices[0].message.content

def run_rag_CoT(model, query):
    retrieved_text = retrieve_relevant_documents(query)
    all_messages = [
        {"role": "system", "content": "You are a helpful tutor who answers questions about a class called Introduction to Deep Learning and LLM based Generative AI Systems"},
        {"role": "user", "content": f"Please answer the following question step-by-step using the given context:\n\n {query}\n\n {"="*50}\n\nCONTEXT: {retrieved_text}\n\n"},
        {"role": "user", "content": f"Remember to break down the question step by step before giving me the final answer"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        max_tokens=1500,
    )
    return response.choices[0].message.content

def run_rag_self_consistency(model, query, num_samples=5):
    self_consistency_responses = []
    for _ in range(num_samples):
        response = run_rag(model, query)
        self_consistency_responses.append(response)

    response_text = "\n".join(
        [f"{i + 1}. {resp}" for i, resp in enumerate(self_consistency_responses)]
    )

    consistency_prompt = f"Please return the text of the best response to the question:\n\n{query}\n\n\n{response_text}"

    evaluation_prompt = [
        {"role": "system", "content": "You are an evaluator for answers to a question about a class called Introduction to Deep Learning and LLM-based Generative AI Systems. Your task is to pick the best response to a question. Return only the text of the best response."},
        {"role": "user", "content": consistency_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=evaluation_prompt,
        max_tokens=1500,
    )
    return response.choices[0].message.content

# Endpoint for handling RAG requests
@app.route('/run-rag', methods=['POST'])
def handle_rag_request():
    try:
        data = request.json
        query = data.get("query", "")
        model_name = data.get("model", gpt_4o_base)
        print(f"Model: {model_name}")
        
        search_type = data.get("search_type", "No Retrieval")

        if search_type == "No Retrieval":
            print("No Retrieval")
            response = run_rag_no_retrieval(model_name, query)  # Fallback if retrieval not needed
        elif search_type == "RAG":
            print("RAG")
            response = run_rag(model_name, query)
        elif search_type == "RAG + CoT":
            print("RAG + CoT")
            response = run_rag_CoT(model_name, query)
        elif search_type == "RAG + Self-Consistency":
            print("RAG + Self-Consistency")
            response = run_rag_self_consistency(model_name, query)
        else:
            return jsonify({"error": "Invalid search type"}), 400

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)