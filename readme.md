Agente RAG: Chatbot Inteligente con Cohere y LangChain



Este proyecto implementa una solución de Retrieval-Augmented Generation (RAG), diseñada para interactuar con documentos PDF locales. El agente está optimizado para responder consultas basándose exclusivamente en la información contenida en dichos documentos, evitando alucinaciones o respuestas fuera de contexto mediante un prompt estricto y el uso de un LLM de alto rendimiento (Cohere).



Arquitectura del Sistema



La solución sigue un flujo de procesamiento de datos optimizado para una alta precisión:



Ingesta: Lectura de archivos PDF mediante PyPDFDirectoryLoader.



Fragmentación: Procesamiento de textos mediante RecursiveCharacterTextSplitter para asegurar que el contexto sea óptimo para la búsqueda.



Embeddings: Conversión de texto a vectores semánticos usando el modelo embed-multilingual-v3.0 de Cohere.



Almacenamiento Vectorial: Indexación local eficiente mediante FAISS.



Recuperación \& Generación:



El sistema recupera los 3 fragmentos más relevantes para la consulta.



Un PromptTemplate estricto instruye al modelo a responder solo basándose en el contexto proporcionado.



Si no se encuentra respuesta, el modelo responde: "No tengo información relacionada a la pregunta en los documentos proporcionados."



Ejemplos de uso



Pregunta



Comportamiento del Agente



"¿Cuál es la política de vacaciones?"



Responde basándose en el contenido de los PDFs cargados.



"¿Quién ganó la última Copa del Mundo?"



Responde: "No tengo información relacionada a la pregunta en los documentos proporcionados."



Instrucciones de Instalación



1\. Clonar el repositorio



git clone <URL\_DE\_TU\_REPOSITORIO>

cd <NOMBRE\_DEL\_PROYECTO>





2\. Configurar el entorno



Crea un archivo .env en la raíz del proyecto y añade tu API Key de Cohere:



COHERE\_API\_KEY="tu\_clave\_api\_aqui"





3\. Instalar dependencias



pip install -r requirements.txt





4\. Ejecutar la aplicación



Coloca tus archivos PDF en la carpeta documentos/ y ejecuta:



streamlit run app.py





⚠️ Nota de Seguridad importante



Para mantener la integridad y privacidad de tus datos:



Nunca subas tus archivos PDF a GitHub.



Asegúrate de que el archivo .gitignore contenga la carpeta documentos/ y el archivo .env.



Archivo .gitignore recomendado:



.env

documentos/

\_\_pycache\_\_/

venv/



