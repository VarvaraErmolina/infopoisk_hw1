from sklearn.feature_extraction.text import CountVectorizer


def build_tf_lib(docs):
    """
    Частотный индекс  с помощью библиотеки scikit-learn
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix
