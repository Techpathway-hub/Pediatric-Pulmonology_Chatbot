import streamlit as st
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util

# Page config
st.set_page_config(page_title="PedsPulmoBot", layout="centered")
st.title("PedsPulmoBot: Ask Me About Pediatric Pulmonary Diseases")
st.sidebar.markdown("👨‍⚕️ This bot is built by Abdulateef, Amaka and Agede")

# Load model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load data and precompute embeddings (cached)
@st.cache_data
def load_data():
    df = pd.read_csv("pediatric_pulmonology_QA_dataset_complete.csv")
    df["embedding"] = df["Question"].apply(lambda x: model.encode(str(x), convert_to_tensor=True))
    return df

faq = load_data()

# Sidebar disease filter
disease_filter = st.sidebar.selectbox("🩺 Filter by Disease", ["All Diseases"] + sorted(faq["Disease"].unique()))

# Apply filter to the dataset
if disease_filter != "All Diseases":
    filtered_faq = faq[faq["Disease"] == disease_filter].copy()
else:
    filtered_faq = faq.copy()

# Friendly answer wrapper
def friendly_wrap(answer):
    openings = [
        "Sure thing! ",
        "Great question. ",
        "Absolutely! ",
        "You got it! ",
    ]
    return random.choice(openings) + answer

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="chat_input")

# Handle input
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Encode user question
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity with stored questions
    similarities = [float(util.cos_sim(user_embedding, emb)) for emb in filtered_faq["embedding"]]
    filtered_faq["similarity"] = similarities
    best_match = filtered_faq.loc[filtered_faq["similarity"].idxmax()]

    # Check confidence threshold
    if best_match["similarity"] > 0.4:
        answer = best_match["Answer"]
        response = friendly_wrap(answer)
    else:
        response = "I'm not confident about that answer. Try asking a more specific question about a disease or topic."

    st.session_state.chat_history.append(("bot", response))

# Display conversation
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**PedsPulmoBot:** {message}")

