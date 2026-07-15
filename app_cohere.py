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
st.set_page_config(
    page_title="Agente RAG - DIGITAL BANK",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para fondo profesional
st.markdown("""
    <style>
    /* Fondo principal */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1f3a5f 0%, #2d5a8c 50%, #1f3a5f 100%) !important;
        min-height: 100vh;
    }
    
    [data-testid="stSidebarContent"] {
        background: #f8f9fa !important;
    }
    
    /* Contenedor principal */
    main {
        background: linear-gradient(135deg, #1f3a5f 0%, #2d5a8c 50%, #1f3a5f 100%) !important;
    }
    
    /* Mensajes del chat */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        background-color: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Título */
    h1 {
        color: white;
        font-weight: 700;
        text-align: center;
        padding: 30px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input del chat */
    .stChatInputContainer {
        border-radius: 10px;
        background-color: white !important;
        padding: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Botones */
    .stButton > button {
        border-radius: 8px;
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #1a5fa0 !important;
    }
    
    /* Cuadro de información */
    div[data-testid="stMarkdownContainer"] > p {
        color: #000000 !important;
    }
    
    div.stAlert {
        background-color: #e8f0f8 !important;
        color: #000000 !important;
        border-left: 4px solid #1f77b4 !important;
    }
    
    div.stAlert p {
        color: #000000 !important;
    }
    
    div.stAlert strong {
        color: #1f3a5f !important;
    }
    
    /* Texto dentro del cuadro de info */
    [data-testid="stMarkdownContainer"] {
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot Experto en Documentos de Politicas de DIGITAL BANK S.A.")

# Cuadro informativo sobre las capacidades del bot
st.info("""
📋 **Sobre este Chatbot**

Este asistente inteligente está entrenado para responder preguntas específicamente sobre las políticas de DIGITAL BANK S.A.

**✅ Puede responder sobre:**
- Política de privacidad y protección de datos
- Términos y condiciones de uso
- Preguntas frecuentes sobre transacciones y límites
- Política de seguridad y prevención de fraudes
- Tarifas y comisiones del servicio

**❌ No puede responder sobre:**
- Información fuera de los documentos cargados
- Preguntas generales no relacionadas con las políticas
- Temas que no estén documentados

💡 **Tip:** Haz preguntas claras y específicas para obtener mejores resultados.
""")

# Cargar variables de entorno
load_dotenv()

# Función con caché para no recargar documentos en cada mensaje
@st.cache_resource
def procesar_documentos():
    ruta = r"documentos"
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
        "Inicia la conversación con un saludo cordial presentandote solo en el primer mensaje y ofrece ayuda adicional al final de tu respuesta. "
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