import feedparser
import pandas as pd
import time
import random
import os
import re
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from urllib.parse import urlparse

# Define RSS feeds for different categories
RSS_FEEDS = {
    'Politics': [
        'http://feeds.bbci.co.uk/news/politics/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml',
        'https://www.theguardian.com/politics/rss',
        'https://www.politico.com/rss/politics.xml'
    ],
    'Business': [
        'http://feeds.bbci.co.uk/news/business/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
        'https://www.theguardian.com/business/rss',
        'https://www.cnbc.com/id/10001147/device/rss/rss.html'
    ],
    'Health': [
        'http://feeds.bbci.co.uk/news/health/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml',
        'https://www.theguardian.com/society/health/rss'
    ]
}

# Function to clean article text
def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    return text

def extract_article_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        time.sleep(random.uniform(1,3))

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        domain = urlparse(url).netloc

        article_text = ""

        if 'bbc' in domain:
            article_elements = soup.select('article p, .ssrcss-uf6wea-RichTextComponentWrapper p')
            article_text = ' '.join([p.get_text() for p in article_elements])
        elif 'nytimes' in domain:
            article_elements = soup.select('section[name="articleBody"] p, .StoryBodyCompanionColumn p')
            article_text = ' '.join([p.get_text() for p in article_elements])
        elif 'theguardian' in domain:
            article_elements = soup.select('.articleBody p, .dcr-1548w16 p')
            article_text = ' '.join([p.get_text() for p in article_elements])
        elif 'cnbc' in domain:
            article_elements = soup.select('.ArticleBody-articleBody p')
            article_text = ' '.join(p.get_text() for p in article_elements)
        elif 'politico' in domain:
            article_elements = soup.select('.story-text p')
            article_text = ' '.join([p.get_text() for p in article_elements])
        
        if not article_text:
            potential_containers = soup.select('article, .article, .post, .content, .main-content, main')

            for container in potential_containers:
                paragraphs = container.select('p')
                if paragraphs:
                    article_text = ' '.join([p.get_text() for p in paragraphs])
                    break
        if not article_text:
            paragraphs = soup.select('p')
            article_text = ' '.join([p.get_text() for p in paragraphs[:15]])
        article_text = clean_text(article_text)

        if len(article_text)> 3000:
            return article_text[:3000] + "..."
        
        return article_text
    
    except Exception as e:
        print(f"Error extracting article text from {url}: {str(e)}")
        return ""
# Function to crawl RSS feeds and collect articles
def crawl_rss_feeds(min_articles_per_category=50, max_articles_per_category=60):
    print("Starting RSS feed crawling...")

    data = {
        "title": [],
        "text": [],
        "category": [],
        "source": [],
        "url": [],
        "date": []
    }

    current_date = datetime.now().strftime("%Y-%m-%d")

    for category, feeds in RSS_FEEDS.items():
        print(f"\nCrawling {category} feeds...")
        articles_count = 0

        for feed_url in feeds:
            if articles_count >= max_articles_per_category:
                break

            try:
                print(f"Parsing feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                domain = urlparse(feed_url).netloc

                for entry in feed.entries:
                    if articles_count >= max_articles_per_category:
                        break

                    article_url = entry.link

                    title = entry.title

                    print(f"Extracting article: {title}")
                    article_text = extract_article_text(article_url)

                    if not article_text or len(article_text.split()) <30:
                        print(f"skipping article (insufficient content): {title}")
                        continue

                    data["title"].append(title)
                    data["text"].append(article_text)
                    data["category"].append(category)
                    data["source"].append(domain)
                    data["url"].append(article_url)
                    data["date"].append(current_date)

                    articles_count += 1

                    print(f"Added article {articles_count}/{max_articles_per_category} for {category}")

            except Exception as e:
                print(f"Error process feed {feed_url}: {str(e)}")

        if articles_count < min_articles_per_category:
            print(f"Waring: Could only collect {articles_count} articles for {category}, which is beow the minimum of {min_articles_per_category}")
    df = pd.DataFrame(data)
    print(f"\nCrawling completed. Collected a total of {len(df)} articles.")


    print("\nArticles per category:")
    print(df['category'].value_counts())

    return df
def save_dataset(df, output_dir="dataset"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    csv_path = os.path.join(output_dir, f"news_articles_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to {csv_path}")

    classifier_df = df[['text', 'category']].copy()
    clasifier_csv_path = os.path.join(output_dir, f"classifier_data_{timestamp}.csv")
    classifier_df.to_csv(clasifier_csv_path, index=False)
    print(f"Simplified dataset for classifier saved to {clasifier_csv_path}")

    return csv_path, clasifier_csv_path
        
if __name__ == "__main__":
    # Crawl the RSS feeds
    df = crawl_rss_feeds()
    
    # Save the collected data
    save_dataset(df)


