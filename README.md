# AI Chatbot Assistant for CV

Welcome to my AI Chatbot Assistant repository! This project is designed to provide a seamless and interactive experience for users seeking information about my professional background and qualifications. 

## Purpose

The primary purpose of this AI-powered chatbot is to assist users by answering any questions related to my CV. Whether you're curious about my work experiences, skills, education, or any other relevant details, this chatbot is here to provide accurate and helpful responses in real-time.

## Features

- **Interactive Dialogue**: Engage with the chatbot in a natural conversational manner.
- **Comprehensive Information**: Get answers to a wide range of questions about my professional history and skills.
- **User-Friendly Interface**: Designed for ease of use, making it accessible for everyone.

Feel free to explore the code, contribute, and make this chatbot even better! Your feedback and suggestions are always welcome.

change it to below structure:

               ┌───────────────────┐
               │   PDF Documents   │
               │  doc1.pdf, doc2.pdf ...
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │   Text Chunking   │
               │  Split PDFs into  │
               │  manageable chunks│
               └─────────┬─────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
   Chunk 1: "Hello world ..."   Chunk 2: "Chroma is great ..."
   Chunk 3: "Test embedding ..." ...

                         │
                         ▼
               ┌───────────────────┐
               │  Embedding Model  │
               │  (OpenAI, SBERT,  │
               │   GoogleGenAI)    │
               └─────────┬─────────┘
                         │
                         ▼
       ┌─────────────────────────────────────┐
       │        Embeddings Generated          │
       │ Chunk 1 → [0.12, 0.56, -0.33,...]  │
       │ Chunk 2 → [0.91, -0.12, 0.44,...]  │
       │ Chunk 3 → [0.22, 0.35, 0.18,...]   │
       └─────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │   ChromaDB        │
               │  Collection: "cv_chunks" │
               ├─────────┬─────────┤
               │ id      │ hash of chunk │
               │ text    │ chunk content │
               │ embedding│ vector       │
               │ metadata │ {"source":"doc1.pdf", "page":1} │
               └─────────┴─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  User Query        │
               │  "Explain main topic" │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ Query Embedding   │
               │ [0.11,0.57,-0.3...]│
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Similarity Search │
               │  Find top-k chunks │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Retrieved Chunks │
               │  (with metadata)  │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  LLM / RAG Model  │
               │  Generates Answer │
               └───────────────────┘
