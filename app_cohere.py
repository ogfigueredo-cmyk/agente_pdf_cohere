import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar las variables de entorno
load_dotenv()

if "COHERE_API_KEY" not in os.environ:
    print("❌ Error: No se encontró la API Key. Ejecuta primero la celda del .env")
else:
    print("⏳ Iniciando el agente y procesando documentos...")

    # 2. Cargar documentos
    loader = PyPDFDirectoryLoader("sample_data/documentos")
    docs = loader.load()

    if not docs:
        print("⚠️ No se encontraron PDFs en la carpeta 'documentos'.")
    else:
        # 3. Dividir texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 4. Crear Base de Datos Vectorial y el Recuperador (Retriever)
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 5. Configurar LLM, Prompt Estricto y Parser
        llm = ChatCohere(model="command-a-plus-05-2026", temperature=0) # Cambiado a "command"

        system_prompt = (
            "Eres un 