import streamlit
import streamlit as st

streamlit.set_page_config(layout='wide')

def main():
    st.title("Similarity Analyzer App")
    st.write("Welcome to the Similarity Analyzer App! This app utilizes Burrows Delta and Cosine Similarity functions to analyze the similarity between texts.")
    st.write("Please select an option on the left and upload your text files to get started!")

if __name__ == "__main__":
    main()
