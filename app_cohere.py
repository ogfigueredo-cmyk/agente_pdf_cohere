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
            "Eres un asistente útil y experto. "
            "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. "
            "Si la respuesta no está en el contexto, responde EXACTAMENTE con esta frase: "
            "'No tengo información relacionada a la pregunta en los documentos proporcionados.' "
            "No intentes inventar una respuesta ni uses tu conocimiento general. "
            "Mantén la respuesta concisa y profesional.\n\n"
            "Contexto:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Cadena simple y moderna (Prompt -> LLM -> Texto limpio)
        cadena_llm = prompt | llm | StrOutputParser()

        print("\n✅ ¡Documentos procesados! El agente está listo.")
        print("👉 (Escribe 'salir' en cualquier momento para terminar el chat)\n")
        print("-" * 50)

        # 6. Bucle de chat con RAG Manual
        while True:
            pregunta = input("👤 Tú: ")

            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("🤖 Agente: ¡Hasta luego!")
                break

            # PASO A: Recuperar los documentos manualmente
            documentos_relacionados = retriever.invoke(pregunta)

            # PASO B: Extraer solo el texto de los documentos encontrados y unirlo
            texto_contexto = "\n\n".join([doc.page_content for doc in documentos_relacionados])

            # PASO C: Enviar el contexto y la pregunta a la cadena
            respuesta = cadena_llm.invoke({
                "context": texto_contexto,
                "input": pregunta
            })

            print(f"🤖 Agente: {respuesta}\n")
