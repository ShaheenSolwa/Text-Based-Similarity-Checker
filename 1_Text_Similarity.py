import pandas as pd
import streamlit as st
import os, math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import string


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Additional preprocessing steps if needed
    return text

def get_cosine_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Create a CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the texts into vectors
    vectors = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity between the vectors
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    return cosine_sim

def burrows_delta_similarity(text1, text2):
    # Step 1: Calculate the word frequencies for each text
    freq1 = calculate_word_frequencies(text1)
    freq2 = calculate_word_frequencies(text2)

    # Step 2: Calculate the document length for each text
    length1 = calculate_document_length(freq1)
    length2 = calculate_document_length(freq2)

    # Step 3: Calculate the similarity score
    score = calculate_similarity_score(freq1, freq2, length1, length2)

    return score


def calculate_word_frequencies(text):
    words = text.lower().split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq


def calculate_document_length(freq):
    length = 0
    for count in freq.values():
        length += count ** 2
    return math.sqrt(length)


def calculate_similarity_score(freq1, freq2, length1, length2):
    score = 0
    for word in set(freq1.keys()).union(set(freq2.keys())):
        count1 = freq1.get(word, 0)
        count2 = freq2.get(word, 0)
        score += (count1 / length1 - count2 / length2) ** 2
    return math.sqrt(score)

def main():
    st.title("Text Similarity Analyzer")
    st.write("Enter two texts and select a similarity method to determine the similarity between them.")

    text1 = st.text_area("Enter Text 1:")
    text2 = st.text_area("Enter Text 2:")

    similarity_method = st.selectbox("Select Similarity Method:", ("Cosine Similarity", "Burrows Delta Similarity"))

    if st.button("Calculate Similarity"):
        if similarity_method == 'Cosine Similarity':
            similarity_score = get_cosine_similarity(text1, text2)

            similarity_table = {'Text 1': [text1], 'Text 2': [text2], 'Similarity Score': [similarity_score]}
            df = pd.DataFrame(similarity_table)
            st.table(df)
        else:
            similarity_score = burrows_delta_similarity(text1, text2)

            similarity_table = {'Text 1': [text1], 'Text 2': [text2], 'Similarity Score': [similarity_score]}
            df = pd.DataFrame(similarity_table)
            st.table(df)


if __name__ == "__main__":
    main()
