import os
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Force CPU to avoid GPU freezes


# Streamlit page config
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better design
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 12em;
        border-radius: 10px;
        border: none;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Cached model loading
@st.cache_resource
def load_model_safe():
    from tensorflow.keras.models import load_model
    model_path = 'simple_rnn_imdb.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

model = load_model_safe()

# Helper functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    if model is None:
        return "Unknown", 0.0
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, float(prediction[0][0])

# Streamlit layout
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.subheader("Classify your movie reviews as Positive or Negative")

# Sidebar info
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Enter a movie review in the text box.
    2. Click 'Analyze Review'.
    3. See sentiment result and prediction score.
    """
)

# User input
user_input = st.text_area("Enter your Movie Review here:", height=150)

# Prediction button and results
if user_input.strip():
    if st.button("Analyze Review"):
        with st.spinner("Analyzing review..."):
            sentiment, score = predict_sentiment(user_input)

            # Color-coded sentiment
            if sentiment == "Positive":
                st.success(f"✅ Sentiment: {sentiment} with score {score}")
            elif sentiment == "Negative":
                st.error(f"❌ Sentiment: {sentiment}")
            else:
                st.warning("⚠️ Model not found!")

            # Dynamic prediction score
            st.write("Prediction Confidence:")
            st.progress(score if sentiment=="Positive" else 1-score)

            # Emoji feedback
            if sentiment == "Positive":
                st.balloons()
            else:
                st.snow()
else:
    st.info("💡 Please enter a movie review above and click 'Analyze Review'.")
