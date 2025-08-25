import os
import yaml
import getpass

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

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
    

class PDFManager:
    """"""
    def __init__(self, pdf_paths, chunk_size, chunk_overlab, top_k) -> None:
        self.pdf_paths = pdf_paths
        self.chunk_size = chunk_size
        self.chunk_overlab= chunk_overlab
        self.top_k= top_k

    def __iter__(self):
        return iter(self.pdf_paths)
    
    @st.cache_resource
    def load_embeddings():
        """"
        grpc.aio (which GoogleGenerativeAIEmbeddings uses internally) needs an active asyncio event loop, 
        but Streamlit runs your script in a thread without one by default.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    @st.cache_resource(show_spinner=False)
    def __call__(self):
        all_docs=[]
        for pdf in self.pdf_paths:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            all_docs.extend(docs)

        splitter= RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlab= self.chunk_overlab,
        )

        chunks = splitter.split_documents(all_docs)

        embedding = PDFManager.load_embeddings()
        vectorstor = FAISS.from_documents(chunks, embedding)

        return vectorstor.as_retriever(
            search_type= 'similarity',
            search_kwargs = {'k': self.top_k}
        )
    

class ChatBotApp:
    def __init__(self, config: ConfigManager) -> None:
        self.config= config
        self.system_prompt= config.get('chat', 'system_prompt', '')

        self.retriever = PDFManager(
            pdf_paths = config.get("data", "pdf_paths", []),
            chunk_size = config.get("rag", "chunk_size", 1000),
            chunk_overlab = config.get("rag", "chunk_overlab", 100),
            top_k = config.get("rag", "top_k", 4)
        )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history= []

    def get_response(self, query, chat_history):
        """ Get response from chatbot """

        output_parser= StrOutputParser()
        
        template ="""
            {self.system_prompt}
            Chat history: {chat_history}
            User question: {question}
            Context: {context}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = GoogleGenerativeAI(
            model= self.config.get('llm', 'model_name', ''),
            temperature= self.config.get('llm', 'temperature', '')
            )

        chain= prompt | llm | output_parser

        try:
            retriever = self.retriever()
            docs = retriever.get_relevant_documents(query)
            context= "\n\n".join(doc.page_content for doc in docs)

            response = chain.invoke(
                {
                    'chat_history': chat_history,
                    'question': query,
                    'context': context
                })
            return response

        except Exception as e:
            st.error(f'Error generating response: {e}')
            return "Sorry, I encountered an error while generating the response."


    def __call__(self):

        LLM_CONTEXT_TURNS= self.config.get('chat', 'history_limit', '')
            
        if not LLM_CONTEXT_TURNS:
            LLM_CONTEXT_TURNS=6

        st.set_page_config(
            page_title= self.config.get('app', 'page_title', 'Hossein Nekouei CV Assistant'),
            page_icon= self.config.get('app', 'page_icon', ':books:'),
            layout= self.config.get('app', 'layout', 'centered')
        )

        st.header(self.config.get('app', 'header', ''))

        # Sidebar cache clear option
        if st.sidebar.checkbox("🔄 Clear Cache"):
            st.cache_resource.clear()
            st.sidebar.success("Cache cleared! Please refresh.")

        # Chat interface
        user_query= st.chat_input("Ask me anything ...")

        for message in st.session_state.chat_history:
            role = 'human' if isinstance(message, HumanMessage) else 'ai_assistant'
            avatar= r'avatar/no_name.png' if role== 'human' else r'avatar/ELIZA.png'
            st.chat_message(role, avatar= avatar).markdown(message.content)

        if user_query and user_query.strip():
            st.session_state.chat_history.appednd(HumanMessage(content=user_query))
            st.chat_message(
                'human', 
                avatar=r'avatar/no_name.png').markdown(user_query)

            chat_history = st.session_state.chat_history[-LLM_CONTEXT_TURNS]
            ai_response = self.get_response(user_query, chat_history)
            ai_message= AIMessage(content= ai_response)

            st.chat_message(
                'assistant', 
                avatar=r'avatar/ELIZA.png').markdown(ai_message.content)
            st.session_state.chat_history.append(ai_message)



def main():
    load_dotenv()

    if not os.environ.get('GOOGLE_API_KEY'):
        os.environ['GOOGLE_API_KEY'] = getpass.getpass('Enter your Google API Key: ')

    config = ConfigManager()
    app = ChatBotApp(config)
    app()


if __name__ == '__main__':
    main()


            