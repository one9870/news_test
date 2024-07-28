import os
import re
from datetime import datetime

def import_news_from_txt(directory_path):
    news_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                date_str = date_match.group(1)
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                date = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            
            title = content.split('\n')[0]
            
            news_data.append({
                'date': date,
                'title': title,
                'content': content
            })
    return news_data