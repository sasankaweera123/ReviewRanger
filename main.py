import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import nltk
import transformers
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.special import softmax
import mysql.connector as mc
import os

sql_database_config = {
    'host': 'localhost',
    'user': os.environ.get('SQL_USERNAME'),
    'password': os.environ.get('SQL_PASSWORD'),
    'database': 'moody'
}

mongo_database_config = {
    'host': 'localhost',
    'port': 27017,
    'username': os.environ.get('MONGO__USERNAME'),
    'password': os.environ.get('MONGO__PASSWORD'),
}


def get_data_from_sql():
    conn = mc.connect(**sql_database_config)

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM reviews")

    data = cursor.fetchall()

    return data


def mongo_data_schema():
    client = pymongo.MongoClient(**mongo_database_config)
    db = client.moody
    collections = db.reviews

    return collections


def main():
    data = get_data_from_sql()
    df = pd.DataFrame(data, columns=['id', 'product_id', 'review_title', 'review_comment', 'rating'])
    print(df.head())

    collections = mongo_data_schema()
    sample_data =[
        {"product_id": "1", "review_title": "title", "review_comment": "comment", "rating": 5},
        {"product_id": "2", "review_title": "title", "review_comment": "comment", "rating": 5},
    ]

    collections.insert_many(sample_data)

    print("Hello World")


if __name__ == '__main__':
    main()
