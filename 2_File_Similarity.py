import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from PyPDF2 import PdfReader
import string, math
import streamlit as st
import pandas as pd

st.set_page_config(layout='wide')

def read_text_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def read_pdf_file(filename):
    with open(filename, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Additional preprocessing steps if needed
    return text

def calculate_cosine_similarity(text1, text2):
    # Preprocess the text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Create a CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the preprocessed text into vectors
    vectors = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

    # Calculate the cosine similarity between the vectors
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    return cosine_sim


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


def read_file(file_path):
    if '.pdf' in file_path.name:
        return read_pdf(file_path)
    else:
        return read_text_file(file_path)


def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


file1 = st.file_uploader('Upload File 1', type=['txt', 'pdf'])
file2 = st.file_uploader('Upload File 2', type=['txt', 'pdf'])

similarity_measure = st.selectbox(
    'Select Similarity Measure',
    ('Cosine Similarity', 'Burrows Delta Similarity')
)


if st.button('Calculate Similarity'):
    if file1 and file2:
        text1 = read_file(file1)
        text2 = read_file(file2)

        if similarity_measure == 'Cosine Similarity':
            similarity_score = calculate_cosine_similarity(text1, text2)
        else:
            similarity_score = burrows_delta_similarity(text1, text2)

        data = {'Similarity Measure': [similarity_measure], 'File 1':[file1.name],
                'File 2':[file2.name], 'Similarity Score': [similarity_score]}
        df = pd.DataFrame(data)
        st.table(df)

    else:
        st.warning('Please upload two files.')


