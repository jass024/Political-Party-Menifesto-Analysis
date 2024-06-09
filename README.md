# Political-Party-Menifesto-Analysis
Overview
This project analyzes political party manifestos using advanced Natural Language Processing (NLP) techniques. The application extracts text from PDF documents, processes the text to identify key themes, and visualizes the results interactively. The goal is to provide insights into the most significant terms, recurring phrases, and thematic similarities across different manifestos.

Features
Text Extraction: Extracts text from PDF documents using PyPDF2.
Preprocessing: Tokenizes text, removes stopwords, and calculates word frequencies using NLTK.
TF-IDF Analysis: Identifies important keywords in each manifesto compared to the entire corpus.
N-gram Analysis: Captures recurring phrases and key promises.
Interactive Visualizations: Uses Plotly and Bokeh for dynamic visualizations, including bar charts and heatmaps.
Similarity Analysis: Computes document similarity using cosine similarity.
Streamlit Integration: Provides a web interface for dynamic exploration of the data.
Installation
Prerequisites
Python 3.11 or higher
Required libraries: nltk, PyPDF2, pandas, plotly, bokeh, scikit-learn, streamlit
