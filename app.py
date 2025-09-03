import os
import re
import yaml
import asyncio
import hashlib
from typing import List, Dict, Any
from pathlib import Path

import streamlit as st
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI


# ---------------- CONFIG MANAGER ----------------
class ConfigManager:
    """Handles loading of yaml configuration."""

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

    def get(self, section, key=None, default=None):
        if key:
            return self.config.get(section, {}).get(key, default)
        
        return self.config.get(section, default)


# ---------------- PDF MANAGER ----------------
class PDFManager:
    """Handles loading and processing of PDF files."""

    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.project_root = Path(__file__).parent

    def load_documents(self):
        documents = []
        
        for pdf in self.pdf_paths:
            pdf_path = (self.project_root / pdf).resolve()
        
            if not pdf_path.exists():
                st.error(f"File not found: {pdf_path}")
                continue
        
            try:
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)

            except Exception as e:
                st.error(f"Error loading PDF {pdf_path}: {e}")
        
        return documents


# ---------------- EMBEDDINGS ----------------
def load_embeddings():

    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        st.error("Google API key not found in environment.")
        st.stop()
    try:
        asyncio.get_running_loop()

    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(api_key)
    )


#-------------------------UTILS--------------------------------
def load_pdf_documents():
    config = ConfigManager()
    pdf_paths = config.get("data", "pdf_paths", [])
    
    if not pdf_paths:
        st.warning("No PDF files provided in config.")
        return []
    
    pdf_manager = PDFManager(pdf_paths)
    return pdf_manager.load_documents()


def clean_duplicates(text):
    # Replace 2+ consecutive same letters with just one
    return re.sub(r'(.)\1+', r'\1', text)
          

def hash_text(text:str)-> str:
    hash_text =hashlib.sha256(text.encode()).hexdigest()
    import pdb
    pdb.set_trace()

    return(hash_text)

def split_documents_into_chunks(documents: List[Any]) -> List[str]:

    config = ConfigManager()
    chunk_size = config.get("rag", "chunk_size", 1000)
    chunk_overlap = config.get("rag", "chunk_overlap", 200)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    texts= ""
    for doc in documents:
            texts += doc.page_content

    clean_text= clean_duplicates(texts)
    chunks = text_splitter.split_text(clean_text)
    
    return chunks

#---------------------VECTOR STORE ------------------------------
class VectorStoreCache:
    """ ChromaDB-based Vector Store cache"""

    def __init__(self, persist_dir='./Chroma_Store'):
        self.persist_dir = persist_dir
        self.embedding_model= load_embeddings()
        self.collection = None

    def load_or_create(self, text_chunks: List[str]):
        # Load Existing ChromaDB or create a new with only missing chunks
        if os.path.exists(self.persist_dir) and os.listdir((self.persist_dir)):
            #load existing collection
            self.collection= Chroma(
                embedding_function= self.embedding_model,
                persist_directory= self.persist_dir
            )

        else:
            try:
            #create a new collection
                self.collection= Chroma.from_texts(
                    texts= text_chunks,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_dir
                    )
                
                self.collection.persist()   
                
                return self.collection
            
            except IndexError:
                st.error("Embedding creation failed. Check your API key or embedding model.")
                st.stop()
            
            except Exception as e:
                st.error(f"Unexpected error while creating FAISS index: {e}")
                st.stop()

        # check with chunks are missing
        existing_ids= set(self.collection.get()['ids'])

        # Filter out chunks that already exist
        new_chunks, new_ids= [],[]
        for chunk in text_chunks:
            chunk_id= hash_text(chunk)
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if new_chunks:
            self.collection.add_texts(
                ids = new_ids,
                texts= new_chunks
                )
            self.collection.persist()


    def similarity_search(self, query: str, k=4):

        if not self.collection:
            raise ValueError('Vecto store not loaded. call create_or_load() first')            

        return self.collection.similarity_search(query, k=k)


def get_response(query, chat_history):
    config = ConfigManager()
    documents = load_pdf_documents()

    if not documents:
        return "No documents available to provide context."

    text_chunks = split_documents_into_chunks(documents)
    vector_cache = VectorStoreCache()
    vector_cache.load_or_create(text_chunks)

    try:
        result_docs = vector_cache.similarity_search(query, k=4)
    
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        st.stop()

    if not result_docs:
        st.warning("No relevant content found for your query.")
        return

    context = "\n\n".join(result_doc.page_content for result_doc in result_docs)

    # Structured answer model prompt
    template = """
    You are a helpful AI assistant. Answer the user questions considering the context and the chat history.
    If you don't know the answer, just say you don't know. Don't try to make up:

    Chat history: {chat_history}
    User question: {question}
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = GoogleGenerativeAI(
        model=config.get('llm', 'model_name', 'models/gemini-2.5-flash'),
        temperature=config.get('llm', 'temperature', 0.2)
    )

    output_parser= StrOutputParser()

    chain = prompt | llm | output_parser

    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "question": query,
            "context": context,
        })

        return response

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating the response."


def display_chat_history(chat_history):
    
    for message in chat_history:

        role = 'human' if isinstance(message, HumanMessage) else 'assistant'
        
        avatar = r'avatar/no_name.png' if role == 'human' else r'avatar/ELIZA.png'
        
        st.chat_message(role, avatar=avatar).markdown(message.content)


def main():
    load_dotenv()
    config = ConfigManager()

    LLM_CONTEXT_TURNS = config.get('chat', 'history_limit', 6) or 6
    
    st.set_page_config(
        page_title=config.get('app', 'page_title', 'CV Assistant'),
        page_icon=config.get('app', 'page_icon', ':books:'),
        layout=config.get('app', 'layout', 'centered')
    )
    
    st.header(config.get('app', 'header', ''))
    
    # Sidebar cache clear option
    if st.sidebar.checkbox("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared! Please refresh.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'welcome_added' not in st.session_state:
        welcome_message = config.get('app', 'welcome_message', '') or "Hello, How can I help you?"
        st.session_state.chat_history.append(AIMessage(content=welcome_message))
        st.session_state.welcome_added = True

    user_query = st.chat_input("Ask me Hossein's CV...", key='question')


    if not user_query:
        display_chat_history(st.session_state.chat_history)
        return
    
    display_chat_history(st.session_state.chat_history)
    
    if user_query and user_query.strip():
    
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    
        st.chat_message('human', avatar=r'avatar/no_name.png').markdown(user_query)
    
        chat_history = st.session_state.chat_history[-LLM_CONTEXT_TURNS:]
    
        ai_response = get_response(user_query, chat_history)
    
        ai_message = AIMessage(content=ai_response or "Sorry, I have no response.")
    
        st.chat_message('assistant', avatar=r'avatar/ELIZA.png').markdown(ai_message.content)
    
        st.session_state.chat_history.append(ai_message)


if __name__ == "__main__":
    main()
