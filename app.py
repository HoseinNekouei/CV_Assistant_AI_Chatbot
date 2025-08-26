import os
import yaml
from pathlib import Path

import streamlit as st
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

# ---------------- CONFIG MANAGER ----------------
class ConfigManager:
    """ Handles loading of yaml configuration """

    def __init__(self, path='config.yaml') -> None:
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            st.error(f'Error loading configuration {e}')
            return {}
        
    def get(self, section, key= None, default= None):
        if key:
            return self.config.get(section, {}).get(key, default)
        return self.config.get(section, default)


@st.cache_resource
def load_embeddings():
    """
    Load Google Generative AI embeddings with proper event loop handling.
    """
    import asyncio
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        st.error("Google API key not found in Streamlit secrets.")
        st.stop()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(api_key)
    )


@st.cache_resource(show_spinner=False)
def build_vector(pdf_paths, chunk_size, chunk_overlap, top_k):
    """Load PDFs, split into chunks, build FAISS retriever."""
    
    if not pdf_paths:
        st.warning("No PDF files provided in config.")
        return None
            
    all_docs=[]

    for pdf in pdf_paths:
        pdf_path = Path(pdf).resolve()
        
        if not pdf_path.exists():
            st.error(f"❌ File not found: {pdf_path}")
            continue

        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_docs.extend(docs)

        except Exception as e:
            (f"Error loading PDF {pdf_path}: {e}")

    if not all_docs:
        st.warning("No documents were loaded from PDFs.")
        return None

    splitter= RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap,
    )

    chunks = splitter.split_documents(all_docs)

    embedding = load_embeddings()
    vectorstor = FAISS.from_documents(chunks, embedding)

    return vectorstor.as_retriever(
        search_type= 'similarity',
        search_kwargs = {'k': top_k}
    )            


def get_response(query, chat_history):
    """Get response from LLM + retriever context."""
    config= ConfigManager()

    pdf_paths = config.get("data", "pdf_paths", [])
    chunk_size = config.get("rag", "chunk_size", 1000)
    chunk_overlap = config.get("rag", "chunk_overlap", 100)
    top_k = config.get("rag", "top_k", 4)
    
    output_parser = StrOutputParser()
    system_prompt = config.get('chat', 'system_prompt', '')

    template = (
        f"{system_prompt}\n"
        "Chat history: {{chat_history}}\n"
        "User question: {{question}}\n"
        "Context: {{context}}\n"
    )

    prompt = ChatPromptTemplate.from_template(template)

    llm = GoogleGenerativeAI(
        model= config.get('llm', 'model_name', 'models/gemini-2.5-flash'),
        temperature= config.get('llm', 'temperature', 0.2)
        )

    chain= prompt | llm | output_parser

    retriever = build_vector(pdf_paths, chunk_size, chunk_overlap, top_k)
    if retriever is None:
        return
    try:
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)

        response = chain.invoke(
            {
                "chat_history": chat_history,
                "question": query,
                "context": context,
            }
        )
        return response

    except Exception as e:
        st.error(f'Error generating response: {e}')
        return "Sorry, I encountered an error while generating the response."


def main():
    load_dotenv()

    config = ConfigManager()
    
        # Initiale session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history= []

        # Add default AI welcome message
        welcome_message = config.get('app', 'welcome_message', '') or 'Hello, How can I help you?'
        ai_welcome= AIMessage(content= welcome_message)
        st.session_state.chat_history.append(ai_welcome)

    LLM_CONTEXT_TURNS= config.get('chat', 'history_limit', 6) or 6

    st.set_page_config(
    page_title= config.get('app', 'page_title', 'Hossein Nekouei CV Assistant'),
    page_icon= config.get('app', 'page_icon', ':books:'),
    layout= config.get('app', 'layout', 'centered')
    )

    st.header(config.get('app', 'header', ''))

    # Sidebar cache clear option
    if st.sidebar.checkbox("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared! Please refresh.")

    # Chat interface
    user_query= st.chat_input("Ask me anything ...")

    for message in st.session_state.chat_history:
        role = 'human' if isinstance(message, HumanMessage) else 'assistant'
        avatar= r'avatar/no_name.png' if role== 'human' else r'avatar/Hossein.jpg'
        st.chat_message(role, avatar= avatar).markdown(message.content)

    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.chat_message(
            'human', 
            avatar=r'avatar/no_name.png').markdown(user_query)

        chat_history = st.session_state.chat_history[-LLM_CONTEXT_TURNS:]
        ai_response = get_response(user_query, chat_history)
        ai_message= AIMessage(content= ai_response if ai_response is not None else "Sorry, I have no response.")

        st.chat_message(
            'assistant', 
            avatar=r'avatar/Hossein.jpg').markdown(ai_message.content)
        st.session_state.chat_history.append(ai_message)


if __name__ == "__main__":
    main()