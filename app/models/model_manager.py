from .news_classifier import PoliticalClassifier, AdClassifier
import joblib

class ModelManager:
    def __init__(self):
        self.political_classifier = PoliticalClassifier()
        self.ad_classifier = AdClassifier()

    def train_political_classifier(self, texts, labels):
        self.political_classifier.train(texts, labels)

    def train_ad_classifier(self, texts, labels):
        self.ad_classifier.train(texts, labels)

    def classify_news(self, text):
        if not self.political_classifier.predict([text])[0]:
            if not self.ad_classifier.predict([text])[0]:
                return "真實事件新聞"
            return "廣告新聞"
        return "政治新聞"

    def save_models(self, political_path, ad_path):
        joblib.dump(self.political_classifier, political_path)
        joblib.dump(self.ad_classifier, ad_path)

    def load_models(self, political_path, ad_path):
        self.political_classifier = joblib.load(political_path)
        self.ad_classifier = joblib.load(ad_path)