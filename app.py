import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.markdown("""
<style>
.stApp { 
    background-color: #f0f8ff; 
    color: #0d47a1 !important; 
}

.stApp p, .stApp span, .stApp label, .stApp li, .stApp div {
    color: #0d47a1 !important;
}

section[data-testid="stSidebar"] { 
    background-color: #e1f5fe !important; 
}
section[data-testid="stSidebar"] * {
    color: #0d47a1 !important;
}

h1, h2, h3, h4, h5, h6 { 
    color: #0288d1 !important; 
}

div.stButton > button {
    background-color: #03a9f4 !important; 
    color: white !important;
    border-radius: 12px;
    padding: 10px 24px;
    border: none;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
}
div.stButton > button * {
    color: white !important;
}
div.stButton > button:hover {
    background-color: #b3e5fc !important; 
    color: #0d47a1 !important;
}

.stTextArea textarea, .stTextInput input {
    background-color: #ffffff !important;
    border-color: #81d4fa !important;
    color: #0d47a1 !important;
}

.streamlit-expanderHeader, .streamlit-expanderContent {
    color: #0d47a1 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌀 Demo TF-IDF en Español")

default_docs = """Amor, no llores, veo luz en tus males
Siguiéndote el corazón, bailando en un canto de zorzales
Niño, soy un hombre con tristeza, sé del peso en tu verdad
Escaparte por robar porque robás para cenar
Vi tus dedos en el barro con olor a libertad
Sé que te querés dormir pa' no volver a despertar."""

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("❄️ Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("🌊 Escribe tu pregunta:", "¿Qué sentimiento transmite la canción?")

with col2:
    st.markdown("### 💎 Preguntas sugeridas:")
    
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Qué sentimiento transmite la canción?"
        st.rerun()
    
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿De qué trata la canción?"
        st.rerun()
        
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Qué emoción expresa?"
        st.rerun()
        
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Cuál es el tema principal?"
        st.rerun()
        
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿La canción habla de tristeza?"
        st.rerun()

if 'question' in st.session_state:
    question = st.session_state.question

if st.button("💎 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        X = vectorizer.fit_transform(documents)
        
        st.markdown("### 🌊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        st.markdown("### ❄️ Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
