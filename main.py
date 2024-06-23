import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Retrieve text files from the current directory
text_files = [file for file in os.listdir() if file.endswith('.txt')]
documents = [open(file, encoding='utf-8').read() for file in text_files]

# Function to vectorize text documents using TF-IDF
def compute_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

# Function to calculate the cosine similarity between two vectors
def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1, vec2])[0][1]

# Vectorize the documents
doc_vectors = compute_tfidf(documents)
file_vectors = list(zip(text_files, doc_vectors))

# Set to store plagiarism results
results = set()

# Function to check for plagiarism between documents
def detect_plagiarism():
    global file_vectors
    for file_a, vec_a in file_vectors:
        comparisons = file_vectors.copy()
        current_index = comparisons.index((file_a, vec_a))
        del comparisons[current_index]
        for file_b, vec_b in comparisons:
            similarity_score = compute_similarity(vec_a, vec_b)
            file_pair = tuple(sorted((file_a, file_b)))
            results.add((file_pair[0], file_pair[1], similarity_score))
    return results

# Print out the plagiarism results
for result in detect_plagiarism():
    print(result)