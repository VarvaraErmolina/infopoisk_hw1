from rank_bm25 import BM25Okapi


def build_bm25_lib(tokenized_docs):
    return BM25Okapi(tokenized_docs)
