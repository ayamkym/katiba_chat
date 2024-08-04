import streamlit as st
import asyncio
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
from langchain.chains import RetrievalQA
import os

# Set up the environment and backend
#os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
persist_directory = "doc_db"
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever()

# Initialize LLM and QA chain
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Set page configuration
st.set_page_config(
    page_title="katiba Chat",
    page_icon="📝",
    layout="centered"  # Use wide layout to ensure full width
)

# Async function to simulate streaming response
async def stream_response(query):
    # Simulate generating a streaming response
    response = qa_chain.invoke({"query": query})
    result = response["result"]
    # For streaming effect, yield chunks of response
    for i in range(0, len(result), 50):  # Adjust chunk size as needed
        yield result[:i+50]
        await asyncio.sleep(0.05)  # Simulate streaming delay

# Main application
def main():
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
                top: 60px;  /* Adjust this value to push the header down */
                left: 0;
                width: 100%;
                z-index: 1000;  /* Ensure it stays above other elements */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                box-sizing: border-box;
            }
            .content {
                margin-top: 140px;  /* Increase this if header height is larger */
                padding: 20px;
            }
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                text-align: center;
                padding: 10px;
               /* background-color: #004d00;*/
                border-top: 1px solid #ccc;
                color: white;
                font-size: 14px;
                z-index: 1000;  /* Ensure it stays above other elements 
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

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_prompt = st.chat_input("Type your message here:")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Placeholder for the assistant response
        assistant_placeholder = st.chat_message("assistant")
        response_placeholder = assistant_placeholder.markdown("...")

        # Asynchronous function to stream response
        async def handle_response():
            async for chunk in stream_response(user_prompt):
                response_placeholder.markdown(chunk)
                st.session_state.chat_history[-1]["content"] = chunk

        asyncio.run(handle_response())

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            © 2024 <a href="https://eleveno'clocklabs.com">eleveno'clock labs</a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
