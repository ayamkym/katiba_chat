import streamlit as st
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set Streamlit page configuration
st.set_page_config(
    page_title="katiba Chat",
    page_icon="📝",
    layout="centered"
)

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize HuggingFace embeddings
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

# Function to load FAISS index
def load_faiss_index(index_path="./doc_db/faiss_index"):
    try:
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectordb
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# Initialize FAISS index
vectordb = load_faiss_index()
if vectordb:
    retriever = vectordb.as_retriever()
else:
    retriever = None

# Define QA prompts
qa_system_prompt = """You are an assistant that provides information strictly based on the Kenyan Constitution document provided to you. \
Use only the following pieces of retrieved context to answer the user’s query. \
If you don't know the answer, just say that you don't know. \
Do not provide information that is outside the provided context or make any assumptions.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Initialize ChatGroq and QA Chain
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
) if retriever else None

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
) if retriever else None

# Function to handle async streaming response
async def stream_response(query):
    if not qa_chain:
        st.error("QA chain is not initialized. Please check your setup.")
        return

    try:
        # Construct the chat history for the prompt
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['chat_history']])
        formatted_query = f"{chat_history}\nUser: {query}"

        # Get the response from the QA chain
        response = qa_chain.invoke({"query": formatted_query})
        result = response["result"]

        # Check if response is within the context
        if "I don't know" in result or "context" in result:
            result = "I'm sorry, I can only provide information based on the Kenyan Constitution."
        
        # Stream the response back to the user
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i:i+chunk_size]
            await asyncio.sleep(0.05)  # Simulate streaming delay
    except Exception as e:
        st.error(f"Error processing query: {e}")

# Main function to run the Streamlit app
def main():
    # Define CSS for the app layout and style
    st.markdown("""
        <style>
            .header {
                background-color: black;
                padding: 20px;
                text-align: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
                position: fixed;
                top: 60px;
                left: 0;
                width: 100%;
                z-index: 1000;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                box-sizing: border-box;
            }
            .content {
                margin-top: 140px;
                padding: 20px;
            }
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                text-align: center;
                padding: 10px;
                border-top: 1px solid #ccc;
                color: white;
                font-size: 14px;
                z-index: 1000;
            }
            .footer a {
                color: white;
                text-decoration: none;
                font-weight: bold;
            }
        </style>
        <div class="header">
            📝 katiba Chat - Ask about the Kenyan Constitution
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Initialize chat history if not already initialized
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input
    user_prompt = st.chat_input("Type your message here:")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Placeholder for the assistant's response
        assistant_placeholder = st.chat_message("assistant")
        response_placeholder = assistant_placeholder.markdown("...")

        # Async handler for the response
        async def handle_response():
            response_text = ""
            async for chunk in stream_response(user_prompt):
                response_text += chunk
                response_placeholder.markdown(response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            
            # Automatically scroll to the bottom of the chat
            st.write('<div id="bottom"></div>', unsafe_allow_html=True)
            st.markdown('<script>document.getElementById("bottom").scrollIntoView();</script>', unsafe_allow_html=True)

        asyncio.run(handle_response())

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer section
    st.markdown("""
        <div class="footer">
            © 2024 <a href="https://eleveno'clocklabs.com">eleveno'clock labs</a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
