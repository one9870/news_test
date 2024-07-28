from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import pandas as pd
from app.models.model_manager import ModelManager
from app.utils.text_processing import import_news_from_txt

app = FastAPI()
model_manager = ModelManager()

class NewsItem(BaseModel):
    date: str
    title: str
    content: str

@app.post("/train")
async def train_models(political_news: List[NewsItem], non_political_news: List[NewsItem], 
                       ad_news: List[NewsItem], real_news: List[NewsItem]):
    political_texts = [item.content for item in political_news + non_political_news]
    political_labels = [1] * len(political_news) + [0] * len(non_political_news)
    model_manager.train_political_classifier(political_texts, political_labels)

    ad_texts = [item.content for item in ad_news + real_news]
    ad_labels = [1] * len(ad_news) + [0] * len(real_news)
    model_manager.train_ad_classifier(ad_texts, ad_labels)

    model_manager.save_models('models/political_model.joblib', 'models/ad_model.joblib')
    return {"message": "Models trained and saved successfully"}

@app.post("/classify_news")
async def classify_news(background_tasks: BackgroundTasks, directory_path: str):
    news_data = import_news_from_txt(directory_path)
    
    classified_news = []
    for i, news in enumerate(news_data, 1):
        classification = model_manager.classify_news(news['content'])
        if classification == "真實事件新聞":
            classified_news.append([i, news['date'], news['title'], classification])
    
    background_tasks.add_task(generate_excel, classified_news)
    
    return {"message": f"Processed {len(news_data)} news articles. Classification completed and Excel file generation started."}

def generate_excel(data):
    df = pd.DataFrame(data, columns=['序號', '日期', '新聞標題', '分類'])
    df.to_excel('classified_news.xlsx', index=False)

if __name__ == "__main__":
    import uvicorn
    model_manager.load_models('models/political_model.joblib', 'models/ad_model.joblib')
    uvicorn.run(app, host="0.0.0.0", port=8000)