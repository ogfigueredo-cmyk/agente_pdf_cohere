import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración de página
st.set_page_config(page_title="Agente RAG", page_icon="🤖")
st.title("🤖 Chatbot Experto en Documentos de Politicas de DIGITAL BANK S.A.")

# Cargar variables de entorno
load_dotenv()

# Función con caché para no recargar documentos en cada mensaje
@st.cache_resource
def procesar_documentos():
    ruta = r"C:\Users\User\Documents\agente_pdf_cohere\documentos"
    if not os.path.exists(ruta):
        return None, None
    
    loader = PyPDFDirectoryLoader(ruta)
    docs = loader.load()
    if not docs:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatCohere(model="command-a-plus-05-2026", temperature=0)
    
    system_prompt = (
        "Eres un asistente útil y experto. "
        "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. "
        "Si la respuesta no está en el contexto, responde EXACTAMENTE con esta frase: "
        "'No tengo información relacionada a la pregunta en los documentos proporcionados.' "
        "No intentes inventar una respuesta ni uses tu conocimiento general. "
        "Inicia la conversación con un saludo cordial presentandote y ofrece ayuda adicional al final de tu respuesta. "
        "Mantén la respuesta concisa y profesional.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    cadena = prompt | llm | StrOutputParser()
    return retriever, cadena

# Inicializar
retriever, cadena_llm = procesar_documentos()

if retriever is None:
    st.error("⚠️ No se encontraron documentos en 'C:\\Users\\User\\Documents\\agente_pdf_cohere\\documentos'. Asegúrate de crear la carpeta y subir los archivos.")
else:
    # Estado del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Recuperar y responder
        docs_relacionados = retriever.invoke(prompt)
        contexto = "\n\n".join([doc.page_content for doc in docs_relacionados])
        
        with st.chat_message("assistant"):
            respuesta = cadena_llm.invoke({"context": contexto, "input": prompt})
            st.markdown(respuesta)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})