# katiba Chat - Kenyan Constitution Chatbot

This is a Streamlit application that provides a chatbot interface to ask questions about the Kenyan Constitution. The chatbot uses `langchain`, `Chroma`, and `ChatGroq` to retrieve and answer questions based on the content of the Kenyan Constitution.

## Features

- **Interactive Chat Interface**: Users can type questions and receive answers about the Kenyan Constitution.
- **Streaming Responses**: The chatbot provides responses in a streaming manner to simulate real-time conversation.
- **Elegant UI**: A clean and appealing user interface with a fixed header and footer, and support for adding images to enhance the UI.

## Requirements

- Python 3.8 or higher
- Streamlit
- asyncio
- langchain_chroma
- langchain_community
- langchain_groq
- os

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/katiba-chat.git
    cd katiba-chat
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your API keys:**

    Create a `secrets.toml` file in the root directory and add your API keys and other sensitive information:

    ```toml
    [api]
    groq_api_key = "your_groq_api_key_here"

    [embeddings]
    model_name = "your_embedding_model_name_here"
    ```

5. **Add `secrets.toml` to `.gitignore`** to prevent it from being pushed to the repository:

    ```gitignore
    secrets.toml
    ```

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run streamlit_app.py
    ```

2. **Interact with the chatbot:**

    Open the app in your browser (typically at `http://localhost:8501`) and start typing your questions about the Kenyan Constitution.

## Project Structure

