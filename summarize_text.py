from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained models and resources
seq2seq_model = load_model("seq2seq_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the TF-IDF Vectorizer and LSA
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("lsa.pkl", "rb") as f:
    lsa = pickle.load(f)

app = Flask(__name__)

MAX_SEQ_LEN = 50

# Function to generate a summary from the seq2seq model
def generate_summary(text):
    input_sequence = tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=MAX_SEQ_LEN)

    summary_prediction = seq2seq_model.predict(input_padded)

    # Convert prediction back to text with meaningful words
    summary = " ".join([tokenizer.index_word[idx] for idx in summary_prediction.argmax(axis=-1).squeeze() if idx > 0])
    return summary

# Function to extract topics from a given text
def extract_topics(text):
    tfidf_vector = tfidf_vectorizer.transform([text])
    topic_scores = np.dot(tfidf_vector.toarray(), lsa.components_.T)

    # Get the top words for the most significant topic
    top_topic_idx = topic_scores.argmax(axis=1)[0]
    terms = tfidf_vectorizer.get_feature_names_out()
    top_indices = lsa.components_[top_topic_idx].argsort()[-5:]  # Top 5 words
    return [terms[idx] for idx in top_indices]

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    paragraph = data.get("paragraph")

    if not paragraph:
        return jsonify({"error": "No paragraph provided"}), 400

    # Generate summary and extract topics
    summary = generate_summary(paragraph)
    topics = extract_topics(paragraph)

    return jsonify({"summary": summary, "topics": topics})

if __name__ == "__main__":
    app.run(debug=True)
