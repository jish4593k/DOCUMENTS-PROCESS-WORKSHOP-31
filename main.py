import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Document collection
documents = [
    "China has a strong economy that is growing at a rapid pace. However, politically it differs greatly from the US Economy.",
    "At last, China seems serious about confronting an endemic problem: domestic violence and corruption.",
    "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people.",
    "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled.",
    "What's the future of Abenomics? We asked Shinzo Abe for his views",
    "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily.",
    "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"
]

# Create a pandas DataFrame
df = pd.DataFrame({'Document': documents})

# Tokenization and preprocessing
df['Tokenized_Document'] = df['Document'].apply(lambda x: x.lower().split())
df['Cleaned_Document'] = df['Tokenized_Document'].apply(lambda x: [word.strip(string.punctuation) for word in x])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Document'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Jaccard Similarity
def jaccard_similarity(doc1, doc2):
    set1 = set(doc1)
    set2 = set(doc2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_similar_documents(query, num_similar=2):
    similarities = [jaccard_similarity(query, doc) for doc in df['Cleaned_Document']]
    sorted_indices = np.argsort(similarities)[::-1][:num_similar]
    similar_docs = df.loc[sorted_indices]
    return similar_docs

def extract_top_keywords(doc, num_keywords=5):
    word_count = Counter(doc)
    top_keywords = word_count.most_common(num_keywords)
    return top_keywords

def compute_tfidf_for_new_document(new_doc):
    new_tfidf_matrix = tfidf_vectorizer.transform([new_doc])
    new_tfidf_df = pd.DataFrame(new_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return new_tfidf_df

# Example usage:

# Find similar documents to a query
query = ["China's economy and politics"]
similar_docs = get_similar_documents(query, num_similar=2)
print("Similar Documents to the Query:")
print(similar_docs)

# Extract the top keywords from a document
doc_index = 2  # Choose a document index
doc_keywords = extract_top_keywords(df['Cleaned_Document'][doc_index])
print("Top Keywords in Document {}:".format(doc_index))
print(doc_keywords)

# Compute TF-IDF for a new document
new_document = "Economy and trade are important aspects of international relations."
new_tfidf = compute_tfidf_for_new_document(new_document)
print("TF-IDF for the New Document:")
print(new_tfidf)
