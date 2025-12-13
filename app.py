import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is downloaded (only run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Configuration and Model Loading (Cached to run once) ---

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_t5_model():
    model_name = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_data
def load_data_and_setup_kmeans():
    # Load necessary dataframes
    df_clustered = pd.read_csv('clustered_sentences.csv')
    sentence_embeddings = np.load('sentence_embeddings.npy')
    df_simplified = pd.read_csv('simplified_sentences.csv')

    # Re-fit KMeans model to get cluster centers
    # Use the same parameters as used in the clustering step
    k = len(df_simplified['cluster_id'].unique()) # Get K from the simplified sentences DataFrame
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(sentence_embeddings)

    # Create mapping from cluster_id to simplified_sentence
    cluster_to_simplified_map = df_simplified.set_index('cluster_id')['simplified_sentence'].to_dict()

    return kmeans_model, cluster_to_simplified_map

# Load models and data
sbert_model = load_sbert_model()
t5_tokenizer, t5_model = load_t5_model()
kmeans_model, cluster_to_simplified_map = load_data_and_setup_kmeans()

# --- Preprocessing Function (from earlier steps) ---
def preprocess_circular_text(text):
    """
    Cleans a single circular's text content.
    - Converts to lowercase.
    - Removes common header/footer/signature patterns (simplified for dummy data).
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove common header/footer/signature patterns
    # These patterns are illustrative and would need to be refined based on actual data.
    text = re.sub(r'further details can be found on our official website\.?'
                  r'|please refer to the updated guidelines on our portal\.?'
                  r'|effective date is \w+ \d{1,2}st?, \d{4}\.?', '', text)
    text = re.sub(r'sincerely,\n.*|best regards,\n.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove multiple spaces and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- Simplification Logic Function ---
def get_simplified_text_for_input(input_raw_text):
    if not input_raw_text:
        return "Please paste some text to simplify."

    # Preprocess the input text
    cleaned_input_text = preprocess_circular_text(input_raw_text)
    if not cleaned_input_text:
        return "Input text became empty after cleaning. Please provide more substantial text."

    # Split into sentences
    input_sentences = nltk.sent_tokenize(cleaned_input_text)

    # Filter empty or very short sentences
    filtered_input_sentences = [s.strip() for s in input_sentences if s.strip() and len(s.split()) >= 5]

    if not filtered_input_sentences:
        return "No valid sentences found after processing. Please provide more content."

    # Generate embeddings for input sentences
    input_sentence_embeddings = sbert_model.encode(filtered_input_sentences)

    simplified_output_sentences = []

    for i, input_embedding in enumerate(input_sentence_embeddings):
        # Find the closest cluster centroid
        similarities_to_centroids = cosine_similarity(input_embedding.reshape(1, -1), kmeans_model.cluster_centers_)[0]
        closest_cluster_id = np.argmax(similarities_to_centroids)

        # Map to its simplified representative sentence
        simplified_sentence = cluster_to_simplified_map.get(closest_cluster_id, "[Simplified sentence not found]")
        simplified_output_sentences.append(simplified_sentence)

    # Reconstruct the simplified circular
    return ". ".join(simplified_output_sentences)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('Official Circular Simplification App')

st.markdown("""
This application simplifies official circulars by breaking them into sentences,
clustering similar sentences, and replacing them with a simplified representative
sentence from that cluster using a T5 model.
""")

user_input = st.text_area("Paste your official circular here:", height=300)

if st.button('Simplify Circular'):
    with st.spinner('Simplifying... This might take a moment.'):
        simplified_text = get_simplified_text_for_input(user_input)
        st.subheader('Simplified Output:')
        st.write(simplified_text)

print("Streamlit application saved to app.py")
