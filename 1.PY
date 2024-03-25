from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Văn bản cần biểu diễn
document = "Nhiều xe phân khối 100-150 cc ở bãi giữ xe cho học sinh ở Hải Phòng"

# Tiền xử lý văn bản
preprocessed_document = document.lower()

# Bag of N-grams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_counts = vectorizer.fit_transform([preprocessed_document])

# TF-IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Biểu diễn dưới dạng numpy array
X_tfidf_array = X_tfidf.toarray()

# In các feature names
print("Vocab:")
print(vectorizer.get_feature_names_out())

# In biểu diễn Bag of N-grams, TF-IDF cho văn bản
print("\nDocument:", document)
print("TF-IDF Vector:", X_tfidf_array[0])
