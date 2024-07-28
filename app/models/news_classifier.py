import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class NewsClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=self.tokenize)),
            ('clf', MultinomialNB())
        ])

    def tokenize(self, text):
        return list(jieba.cut(text))

    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)

    def predict(self, texts):
        return self.pipeline.predict(texts)

class PoliticalClassifier(NewsClassifier):
    pass

class AdClassifier(NewsClassifier):
    pass