import math


def build_tf_index(tokenized_docs):
    """
    Строим обратный индекс (inverted index) с частотами терминов.
    Структура индекса:
       term -> {doc_id: tf}
    """
    index = {}
    for doc_id, tokens in enumerate(tokenized_docs):
        for t in tokens:
            index.setdefault(t, {})
            index[t][doc_id] = index[t].get(doc_id, 0) + 1
    return index


def compute_idf(tf_index, n_docs):
    """
    Вычисляем IDF для всех терминов.
    """
    idf = {}
    for term, posting in tf_index.items():
        df = len(posting)
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    return idf
