
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
from textblob import TextBlob
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation

def get_manifesto_paths(config_file="config.ini"):  
  """
  Reads manifesto file paths from a configuration file.
  If no config file is specified, returns hardcoded paths.
  """

  return {
      "BJP": "C:/Users/HP/Desktop/Election_sentiment/Modi-Ki-Guarantee-Sankalp-Patra-English_2.pdf",
      "Congress": "C:/Users/HP/Desktop/Election_sentiment/Congress_Manifesto_English_d86007236c.pdf",
  }

def extract_text_from_pdf(filepath):
  text = ""
  try:
    with open(filepath, 'rb') as file:
      reader = PyPDF2.PdfReader(file)
      for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
          text += page_text
  except Exception as e:
    print(f"Error extracting text from {filepath}: {e}")
  return text

def process_manifesto(text):
  tokens = word_tokenize(text.lower())
  tokens = [word for word in tokens if word.isalnum()]
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]

  word_counts = Counter(tokens)
  return tokens, word_counts

def tfidf_analysis(corpus):
  vectorizer = TfidfVectorizer(stop_words='english')
  X = vectorizer.fit_transform(corpus)
  feature_names = vectorizer.get_feature_names_out()
  df_tfidf = pd.DataFrame(X.T.toarray(), index=feature_names)
  return df_tfidf

def ngram_analysis(corpus, n=2):
  vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
  X = vectorizer.fit_transform(corpus)
  ngrams = vectorizer.get_feature_names_out()
  ngram_counts = X.sum(axis=0).A1
  ngram_freq = dict(zip(ngrams, ngram_counts))
  return ngram_freq

def sentiment_analysis(text):
  blob = TextBlob(text)
  return blob.sentiment.polarity, blob.sentiment.subjectivity

def topic_modeling(corpus, n_topics=5):
  vectorizer = CountVectorizer(stop_words='english')
  X = vectorizer.fit_transform(corpus)
  lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
  lda.fit(X)
  feature_names = vectorizer.get_feature_names_out()
  return lda, feature_names

def display_topics(model, feature_names, no_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print(f"Topic {topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Process each manifesto and store data
manifesto_paths = get_manifesto_paths()  # Use function to get paths
manifesto_data = {}
for party, filepath in manifesto_paths.items():
  try:
    text = extract_text_from_pdf(filepath)
    tokens, word_counts = process_manifesto(text)
    if tokens:  # Ensure there are valid tokens
      manifesto_data[party] = {
          "text": text,
          "tokens": tokens,
          "word_counts": word_counts
      }
  except Exception as e:
        print(f"Error processing {party} manifesto: {e}")

# Perform TF-IDF analysis
corpus = [data["text"] for data in manifesto_data.values() if data["text"].strip()]
if corpus:  # Ensure corpus is not empty
  df_tfidf = tfidf_analysis(corpus)
  df_tfidf.columns = [party for party in manifesto_data.keys() if manifesto_data[party]["text"].strip()]

  # Display top 10 words by TF-IDF score for each party
  for party in manifesto_data.keys():
    if manifesto_data[party]["text"].strip():
      top_10_tfidf = df_tfidf[party].sort_values(ascending=False).head(10)
      print(f"\nTop 10 TF-IDF words for {party}:")
      print(top_10_tfidf)
else:
  print("Error: Corpus is empty after preprocessing.")

# Generate word clouds
for party, data in manifesto_data.items():
  if data["word_counts"]:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(data["word_counts"])
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {party}")
    plt.show()
  else:
    print(f"Warning: No word cloud generated for {party} due to lack of data.")

# Perform N-gram analysis for each party
for party, data in manifesto_data.items():
  if data["text"].strip():
    bigrams = ngram_analysis([data["text"]], n=2)
    top_10_bigrams = dict(sorted(bigrams.items(), key=lambda item: item[1], reverse=True)[:10])
    print(f"\nTop 10 Bigrams for {party}:")
    print(top_10_bigrams)

# Sentiment analysis for each party
for party, data in manifesto_data.items():
  if data["text"].strip():
    polarity, subjectivity = sentiment_analysis(data["text"])
    print(f"\nSentiment for {party}:")
    print(f"Polarity: {polarity}, Subjectivity: {subjectivity}")

# Topic modeling (uncomment if desired)
#corpus = [data["text"] for data in manifesto_data.values() if data["text"].strip()]
#if corpus:
 #  lda_model, feature_names = topic_modeling(corpus, n_topics=5)
  # display_topics(lda_model, feature_names, 10)

# Streamlit dashboard
st.title("Election Manifesto Analysis")

for party, data in manifesto_data.items():
  if data["word_counts"]:
    st.header(f"{party} Manifesto")

    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(data["word_counts"])
    plt.figure(figsize=(10, 5))  # Create a Matplotlib figure
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


    st.subheader("Top 10 TF-IDF Words")
    if manifesto_data[party]["text"].strip():
      top_10_tfidf = df_tfidf[party].sort_values(ascending=False).head(10)
      st.bar_chart(top_10_tfidf)

    st.subheader("Sentiment Analysis")
    polarity, subjectivity = sentiment_analysis(data["text"])
    st.write(f"Polarity: {polarity}, Subjectivity: {subjectivity}")

    st.subheader("Top 10 Bigrams")
    bigrams = ngram_analysis([data["text"]], n=2)
    top_10_bigrams = dict(sorted(bigrams.items(), key=lambda item: item[1], reverse=True)[:10])
    st.write(top_10_bigrams)

