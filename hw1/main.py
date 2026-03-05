from search import run_search

if __name__ == "__main__":

    results = run_search(
        query="вакцинация",  # текст запроса
        csv_path="corpus.csv",
        method="bm25_lib")  # tf_lib / bm25_lib / tf_dict / bm25_dict / tf_matrix / bm25_matrix

    for score, text in results:
        print(f"{score:.4f} | {text}")
