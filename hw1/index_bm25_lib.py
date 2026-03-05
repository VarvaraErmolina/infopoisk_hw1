from rank_bm25 import BM25Okapi


def build_bm25_lib(tokenized_docs):
    """
    BM-25 индекс  с помощью библиотеки rank_bm25
    """
    return BM25Okapi(tokenized_docs)
