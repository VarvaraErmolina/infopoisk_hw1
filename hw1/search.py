import numpy as np

from data_loader import load_corpus
from preprocessing import preprocess, preprocess_corpus
from index_dict import build_tf_index, compute_idf
from index_matrix import build_vocab, build_tf_matrix, compute_idf_vector
from index_bm25_lib import build_bm25_lib
from index_tf_lib import build_tf_lib


def search_tf_lib(query, vectorizer, matrix, docs, top_k=5):
    # частотный индекс с библиотекой
    q_vec = vectorizer.transform([query])
    scores = (matrix @ q_vec.T).toarray().ravel()
    best = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), docs[i]) for i in best if scores[i] > 0]


def search_bm25_lib(query, bm25, docs, top_k=5):
    # BM-25 индекс с библиотекой
    q = set(preprocess(query))
    scores = bm25.get_scores(q)
    best = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), docs[int(i)]) for i in best if scores[i] > 0]


def search_tf_dict(query, tf_index, docs, top_k=5):
    # частотный индекс руками через словари
    q = preprocess(query)
    scores = {}
    for t in q:
        for doc_id, tf in tf_index.get(t, {}).items():
            scores[doc_id] = scores.get(doc_id, 0) + tf
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(float(score), docs[doc_id]) for doc_id, score in ranked]


def search_bm25_dict(query, tf_index, idf, tokenized_docs, docs, top_k=5):
    # BM-25 индекс руками через словари
    q = preprocess(query)
    k1, b = 1.5, 0.75
    dl = [len(d) for d in tokenized_docs]
    avgdl = sum(dl) / len(dl)

    scores = {}
    for t in set(q):
        post = tf_index.get(t)
        if not post:
            continue
        for doc_id, tf in post.items():
            denom = tf + k1 * (1 - b + b * dl[doc_id] / avgdl)
            s = idf.get(t, 0.0) * (tf * (k1 + 1)) / denom
            scores[doc_id] = scores.get(doc_id, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(float(score), docs[doc_id]) for doc_id, score in ranked]


def _query_vec(query, vocab):
    """
    Преобразуем запрос в вектор терминов.
    Каждый элемент вектора соответствует слову словаря.
    Значение -- частота слова в запросе.
    """
    q_tokens = preprocess(query)
    q = np.zeros(len(vocab), dtype=np.float32)
    for t in q_tokens:
        if t in vocab:
            q[vocab[t]] += 1.0
    return q, q_tokens


def search_tf_matrix(query, tf_matrix, vocab, docs, top_k=5):
    # частотный индекс через матрицы
    q, _ = _query_vec(query, vocab)
    scores = tf_matrix @ q
    best = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), docs[int(i)]) for i in best if scores[i] > 0]


def search_bm25_matrix(query, tf_matrix, vocab, docs, top_k=5):
    # BM-25 индекс через матрицы
    q_vec, q_tokens = _query_vec(query, vocab)
    idf = compute_idf_vector(tf_matrix)

    dl = tf_matrix.sum(axis=1)
    avgdl = float(dl.mean())
    k1, b = 1.5, 0.75

    scores = np.zeros(tf_matrix.shape[0], dtype=np.float32)

    for t in set(q_tokens):
        if t not in vocab:
            continue
        j = vocab[t]
        tf = tf_matrix[:, j]
        denom = tf + k1 * (1 - b + b * dl / avgdl)
        scores += idf[j] * (tf * (k1 + 1)) / denom

    best = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), docs[int(i)]) for i in best if scores[i] > 0]


def run_search(
        query,
        csv_path,
        method="bm25_dict",
        top_k=5,
):
    docs = load_corpus(csv_path)
    tokenized_docs = preprocess_corpus(docs)

    if method == "tf_lib":
        vectorizer, matrix = build_tf_lib(docs)
        return search_tf_lib(query, vectorizer, matrix, docs, top_k)

    if method == "bm25_lib":
        bm25 = build_bm25_lib(tokenized_docs)
        return search_bm25_lib(query, bm25, docs, top_k)

    if method == "tf_dict":
        tf_index = build_tf_index(tokenized_docs)
        return search_tf_dict(query, tf_index, docs, top_k)

    if method == "bm25_dict":
        tf_index = build_tf_index(tokenized_docs)
        idf = compute_idf(tf_index, len(docs))
        return search_bm25_dict(query, tf_index, idf, tokenized_docs, docs, top_k)

    if method == "tf_matrix":
        vocab = build_vocab(tokenized_docs)
        x = build_tf_matrix(tokenized_docs, vocab)
        return search_tf_matrix(query, x, vocab, docs, top_k)

    if method == "bm25_matrix":
        vocab = build_vocab(tokenized_docs)
        x = build_tf_matrix(tokenized_docs, vocab)
        return search_bm25_matrix(query, x, vocab, docs, top_k)

    raise ValueError(f"Unknown method: {method}")
