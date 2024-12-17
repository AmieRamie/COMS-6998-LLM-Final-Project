# Data Augmentation
Data scarcity is a significant barrier to fine-tuning models for most courses. This course, for example, has only a dozen to two dozen files to act as base context. To overcome this, we developed a methodology to augment existing course material with additional context

## Process

1. **Topic Mining from Syllabus:** The course syllabus was processed using GPT-4o to extract key topics. Some extracted topics include: “synchronous vs asynchronous SGD,” “motivation and use of attention in LLMs,” and “serving systems for fine-tuned models.” These topics provide a focused framework for identifying relevant external resources.
2. **Augmenting with Blog Posts and Research Articles:** Using a SerpAPI, blog posts and research articles related to each of the mined topics were retrieved from Google.
3. **Ingesting Class Presentation Data:** All class presentation PDFs were parsed to extract linked blog posts and research articles, creating a secondary set of data.
4. **Combining Datasets:** The retrieved and ingested data were combined into a single, comprehensive dataset to augment the limited base material available for the course.
5. **Preparation for Fine-Tuning and RAG:** This augmented dataset was used both to generate synthetic Q/A pairs for fine-tuning and to populate a vector database for RAG.

## Prerequisites

1. Python 3.8 or higher
2. Google Document AI API access
3. OpenAI key for extracting topics

## Code
Sections 1-4 of Extracting Text from PDFs.ipynb walks through how to do this

# Creating a Synthetic Dataset
To generate synthetic Q/A pairs, we used the augmented context and primary course data. We focused on building an open answer quiz Q/A dataset to aim to make the system better at answering questions students would likely face during the time in their respective courses. Hypothetically, we could have prompted the synthetic data creation LLM to create multiple choice Q/As or prompted to make longer, more open ended Q/A’s. In future iterations of this we would like to implement that. Here is the process we implemented:

## Process

1. **Scraping Data from Secondary Data:** The text from every blog post and research article link was scraped
2. **Document Processing:** Each document was passed into GPT-4o, along with prompts specifying the course context, key concepts, and desired Q/A format (quiz questions and answers). The prompt instructed the model to “create up to 10 open-ended quiz questions and answers using the data provided.”
3. **Output Generation:** GPT-4o generated Q/A pairs for each document. These were split into training and test sets.
4. **Dataset Composition:** The final dataset consisted of about 3,700 training examples and 2,500 test examples, following a 60/40 train-test split. This dataset was designed to emulate the types of questions students might encounter or be tested on during their course.

## Prerequisites

1. Python 3.8 or higher
2. requests package
3. OpenAI key for creating synthetic Q/A pairs

## Code
Sections 5 & 7 of Extracting Text from PDFs.ipynb walks through how to do this

<img width="1066" alt="image" src="https://github.com/user-attachments/assets/d366963b-68e2-4853-822a-016edaa0a387" />
### Process of gathering secondary data and building a synthetic dataset

#Fine-Tuning Process
For the Q/A generation process we wanted to use a larger, complex model to get the best synthetic Q/A pairs. For the base model, and the model we wanted to fine-tune we wanted to use a lightweight and cheaper model, to make the economics of tuning and inference more viable for commercial use. Due to this we ended up fine-tuning GPT-4o mini using the OpenAI api. We could have gone with any pretrained model, but we went with this specific one because of its cost effectiveness and OpenAI’s API simplicity. Furthermore, because we were able to generate a large number of synthetic Q/A training examples, we wanted a fine-tuning API that would allow us to do full fine-tuning cheaply. In future iterations, it may be worth seeing how a PEFT model would have performed. Here is the process we used:

## Process

1. **Data Formatting:** The synthetic training Q/A pairs were formatted according to OpenAI’s API requirements.
2. **Model Training:** The formatted JSON file was uploaded to OpenAI’s fine-tuning API to train the GPT-4o mini model.
3. **Inference:** The fine-tuned model was deployed on OpenAI’s servers, where it was used for downstream tasks.

## Prerequisites

1. Python 3.8 or higher
2. OpenAI key for hosting training data and fine tuning model

## Code
Sections 1-2 of Fine-tuning Open AI Model.ipynb walks through how to do this

#Creating RAG Data
To enhance contextual accuracy, we implemented a RAG pipeline using the augmented  secondary blog post/research article dataset (that we also used to create the synthetic Q/A data set):

## Process

1. **Chunking and Embedding:** All blog posts and research articles were chunked into smaller segments and embedded using OpenAI’s text-embedding-3-large model.
2. **Vector Database:** The embeddings were stored in a local vector database for efficient retrieval.
3. **Query Processing and Inference:** During inference, user queries were processed to retrieve the 15 most similar chunks from the vector database. These chunks, along with the query and original context, were concatenated and passed into either the baseline GPT-4o mini model or the fine-tuned version for response generation.

## Prerequisites

1. Python 3.8 or higher

## Code
Sections 6 of Extracting Text from PDFs.ipynb walks through how to do this

#Evaluation Methodology
Two metrics were employed to evaluate system performance. In order to get a deeper understanding of how semantically similar the system variant outputs were as compared to the expected output in the synthetic Q/A test set we employed an embedding similarity metric. We also evaluated each variant using a traditional ROUGE-L Sum score.

## Process

1. Embedding Similarity:
      a. For each test example, the question was passed to a system variant to generate a predicted answer.
      b. Both the predicted answer and the expected answer were embedded using text-embedding-3-large.
      c. Cosine similarity was calculated between the two embeddings to measure semantic alignment.
3. ROUGE-L Sum:
      a. ROUGE-L scores were calculated between the predicted and expected answers.
      b. Scores were summed across all test examples to assess overall alignment.

## Prerequisites

1. Python 3.8 or higher
2. OpenAI key for getting predicted outputs from system variant

## Code
Evaluating Fine TUned Model.ipynb walks through how to set up the evlauation system. Only the non-RAG variants were tested in this notebook, but you can plug any system variant into sections 4-6

<img width="1038" alt="image" src="https://github.com/user-attachments/assets/1f536402-53cd-4780-acc5-a7e7ed080243" />

# Retrieval-Augmented Generation Chat Application

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

## Loading Documents in ChromaDB
The script below will load and preprocess JSON files from the *RAG_data* directory:

```python
import os
import json

def load_json_files(directory):
    """
    Load all JSON files from the specified directory.

    Args:
        directory (str): The path to the directory containing the JSON files.

    Returns:
        list: A list of dictionaries containing the data from all JSON files.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    return data

def extract_text(data):
    """
    Extract text data from the JSON structure, supporting two formats.

    Args:
        data (list): A list of dictionaries containing JSON data.

    Returns:
        list: A list of dictionaries with URLs or file paths and their corresponding text chunks.
    """
    documents = []
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, list):  # First format with URL keys
                for chunk in value:
                    documents.append({"source": key, "text": chunk})
            elif isinstance(value, dict):  # Second format with file paths as keys
                if "text" in value:
                    documents.append({"source": key, "text": value["text"]})
    return documents

# Directory containing the JSON files
directory = "RAG_data"

# Load and preprocess data
json_data = load_json_files(directory)
documents = extract_text(json_data)
```

Next, we will generate embeddings for each document using the OpenAI *text-embedding-3-large* model:
```python
import json
import os
import openai
import timeit
import time
import pandas as pd
import numpy as np
from tiktoken import encoding_for_model
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI()

def get_embedding(text, model="text-embedding-3-large", max_tokens=8192):
    tokenizer = encoding_for_model(model)  
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        print(f"Truncating text to {max_tokens} tokens: {text[:100]}...")  # Log truncation
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Failed to encode text: {e}")
        return None

def get_embeddings(texts, model="text-embedding-3-large", max_tokens=8192, batch_size=100):
    embeddings = []
    tokenizer = encoding_for_model(model)  # Get the tokenizer for the model
   
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Truncate texts in the batch that exceed the token limit
        truncated_batch = []
        for text in batch:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_tokens:
                print(f"Truncating text to {max_tokens} tokens: {text[:100]}...")  # Log truncation
                tokens = tokens[:max_tokens]
                text = tokenizer.decode(tokens)
            truncated_batch.append(text)

        try:
            # Generate embeddings for the batch
            response = client.embeddings.create(input=truncated_batch, model=model)
            batch_embeddings = [item.embedding for item in response.data]
            if i % 5000 == 0:
                print(f"Generated embeddings for batch {i}-{i + batch_size}")
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Unexpected error for batch {i}-{i + batch_size}: {e}")
            embeddings.extend([None] * len(batch))  # Append None for unexpected errors
        
    return embeddings

texts = [doc["text"] for doc in documents]
embeddings = get_embeddings(texts, batch_size=1000)
```

We're now ready to load the documents and embeddings into ChromaDB. This script will create a local SQLite3 database in the *chroma/* directory and load the documents into the 'llm_tutor_collection' collection:
```python
import chromadb
from chromadb.config import Settings
persistent_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
collection = persistent_client.get_or_create_collection("llm_tutor_collection")

batch_size = 500
batch_ids = []
batch_texts = []
batch_metadata = []
batch_embeddings = []

id = 0
for doc, embedding in zip(documents, embeddings):
    # Add data to the current batch
    batch_ids.append(str(id))
    batch_texts.append(doc["text"])
    batch_metadata.append({"source": doc["source"]})
    batch_embeddings.append(embedding)
    id += 1

    # Check if the batch is ready for uploading
    if len(batch_ids) == batch_size:
        # Upload the batch to ChromaDB
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata,
            embeddings=batch_embeddings
        )
        # Clear the batch lists
        batch_ids = []
        batch_texts = []
        batch_metadata = []
        batch_embeddings = []

# Upload any remaining data in the last batch
if batch_ids:
    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        metadatas=batch_metadata,
        embeddings=batch_embeddings
    )
```

## Querying Vector Database
The following method can be used to retrieve documents from the llm_tutor_collection based on a given query:
```python
def retrieve_relevant_documents(query, n_results=5):
    """
    Retrieve the most relevant documents from the ChromaDB vector store.
    
    Args:
        query (str): The user's question or query.
        collection (Collection): The ChromaDB collection object.
        n_results (int): Number of results to retrieve.
    
    Returns:
        str: Concatenated text of the top retrieved documents.
    """
    # Generate embedding for the query using Gemini model
    query_embedding = get_embedding(query)

    # Retrieve top documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Combine text from the retrieved documents
    retrieved_text = " ".join(doc[0] for doc in results["documents"])
    return retrieved_text
```

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
