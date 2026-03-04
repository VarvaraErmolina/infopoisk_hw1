import numpy as np


def build_vocab(tokenized_docs):
    """
    Строим словарь терминов.
    Каждому уникальному слову из корпуса присваивается индекс для построения матрицы документ–термин.
    """
    vocab = {}
    for doc in tokenized_docs:
        for w in doc:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def build_tf_matrix(tokenized_docs, vocab):
    """
    Строим TF-матрицу.

    Строки -- документы
    Столбцы -- термины словаря
    Значения -- частота термина в документе
    """
    d = len(tokenized_docs)
    v = len(vocab)
    x = np.zeros((d, v), dtype=np.float32)
    for doc_id, doc in enumerate(tokenized_docs):
        for w in doc:
            x[doc_id, vocab[w]] += 1.0
    return x


def compute_idf_vector(tf_matrix):
    """
    Вычисляем IDF для всех терминов.
    """
    d = tf_matrix.shape[0]
    df = np.count_nonzero(tf_matrix, axis=0)
    return np.log((d - df + 0.5) / (df + 0.5) + 1)
