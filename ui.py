import streamlit as st
import requests

# Streamlit app configuration
st.set_page_config(page_title="TA LLM", layout="wide")

# API URL for the Flask server
API_URL = "http://localhost:5000/run-rag"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "is_loading" not in st.session_state:
    st.session_state["is_loading"] = False

# Sidebar options for model and search type
st.sidebar.title("RAG Chat Options")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o-mini", "ft:gpt-4o-mini-2024-07-18:f-prime-capital::AbZYSjIT"],
    index=0
)
search_type = st.sidebar.selectbox(
    "Select Search Type",
    ["No Retrieval", "RAG", "RAG + CoT", "RAG + Self-Consistency"],
    index=1
)

# Main chat interface
st.title("TA LLM")
st.write("Let me know if you have any questions about Introduction to Deep Learning and LLM based Generative AI Systems!")

# Display chat history
chat_container = st.container()
with chat_container:
    st.subheader("Chat")
    for i, entry in enumerate(st.session_state["chat_history"]):
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['message']}")
        elif entry["role"] == "assistant":
            st.markdown(f"**TA LLM:** {entry['message']}")

# Input for user queries
with st.form("chat_form"):
    user_query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Send")

# Handle form submission
if submit_button and user_query.strip():
    # Add user query to chat history
    st.session_state["chat_history"].append({"role": "user", "message": user_query})

    # Show loading indicator
    st.session_state["is_loading"] = True
    with st.spinner("Generating response..."):
        # Send request to the Flask server
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": user_query,
                    "model": model_name,
                    "search_type": search_type
                }
            )
            response_data = response.json()
            assistant_response = response_data.get("response", "Error: No response received.")

            # Add assistant response to chat history
            st.session_state["chat_history"].append({"role": "assistant", "message": assistant_response})

            # Rerun to update UI with the latest chat
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Hide loading indicator
            st.session_state["is_loading"] = False

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state["chat_history"] = []
    st.rerun()