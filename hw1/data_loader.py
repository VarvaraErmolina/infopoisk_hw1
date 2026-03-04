import pandas as pd


def load_corpus(csv_path):
    df = pd.read_csv(csv_path, encoding="cp1251", sep=';')

    docs = df["sentence"].dropna().astype(str).tolist()

    return docs
