import re
import nltk
import pymorphy3
from nltk.corpus import stopwords

nltk.download('stopwords')

# для удаления тегов из корпуса вида [saver_san] (типа ников с форумов, которых довольно много в корпусе)
_TAG_RE = re.compile(r"\[[^\]]+\]")
_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)

# для извлечения русских слов (берем только кириллицу)
_morph = pymorphy3.MorphAnalyzer()


def preprocess(text):
    """
    Предобработка:
    - удаление тегов
    - приведение к нижнему регистру
    - токенизация
    - лемматизация
    - удаление стоп-слов
    """
    stopwords_ru = set(stopwords.words("russian"))

    text = _TAG_RE.sub(" ", text).lower()
    tokens = _WORD_RE.findall(text)

    lemmas = []
    for w in tokens:
        lemma = _morph.parse(w)[0].normal_form
        if lemma not in stopwords_ru and len(lemma) > 2:
            lemmas.append(lemma)
    return lemmas


def preprocess_corpus(docs):
    return [preprocess(d) for d in docs]
