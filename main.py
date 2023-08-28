import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import nltk
import transformers
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm
from scipy.special import softmax
import mysql.connector as mc
import os

pd.options.mode.chained_assignment = None

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


def create_data_set():
    data = get_data_from_sql()
    data_set = pd.DataFrame(data, columns=['id', 'product_id', 'review_title', 'review_comment', 'rating'])
    return data_set


def data_cleaning(data_set):
    data_set['review_title'] = data_set['review_title'].str.lower()
    data_set['review_comment'] = data_set['review_comment'].str.lower()
    data_set['review_title'] = data_set['review_title'].str.replace('[^\w\s]', '')
    data_set['review_comment'] = data_set['review_comment'].str.replace('[^\w\s]', '')
    data_set['review_title'] = data_set['review_title'].str.replace('\d+', '')
    data_set['review_comment'] = data_set['review_comment'].str.replace('\d+', '')
    data_set = data_set[(data_set['review_title'].str.strip() != '') | (data_set['review_comment'].str.strip() != '')]
    return data_set


def calculate_sentiment_textblob(data_set):
    data_set['review_title_polarity'] = data_set['review_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data_set['review_title_subjectivity'] = data_set['review_title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    data_set['review_comment_polarity'] = data_set['review_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data_set['review_comment_subjectivity'] = data_set['review_comment'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)

    df = pd.DataFrame()

    if data_set['product_id'].nunique() > 1:
        df['rtp_textblob'] = data_set.groupby('product_id')['review_title_polarity'].mean()
        df['rts_textblob'] = data_set.groupby('product_id')['review_title_subjectivity'].mean()
        df['rcp_textblob'] = data_set.groupby('product_id')['review_comment_polarity'].mean()
        df['rcs_textblob'] = data_set.groupby('product_id')['review_comment_subjectivity'].mean()
    return df


def calculate_sentiment_vader(data_set):
    df = pd.DataFrame()
    sid = SentimentIntensityAnalyzer()
    data_set['review_title_polarity'] = data_set['review_title'].apply(lambda x: sid.polarity_scores(x)['compound'])
    data_set['review_comment_polarity'] = data_set['review_comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

    if data_set['product_id'].nunique() > 1:
        df['rtp_vader'] = data_set.groupby('product_id')['review_title_polarity'].mean()
        df['rcp_vader'] = data_set.groupby('product_id')['review_comment_polarity'].mean()
    return df


def calculate_sentiment_bert(data_set):
    df = pd.DataFrame()
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        classifier = pipeline("sentiment-analysis", model=model_name)

        with tqdm(total=len(data_set), desc="Processing review titles") as pbar:
            review_title_polarity = []
            for review_title in data_set['review_title']:
                polarity = classifier(review_title)[0]['score']
                review_title_polarity.append(polarity)
                pbar.update(1)  # Update progress bar

            data_set['review_title_polarity'] = review_title_polarity

        with tqdm(total=len(data_set), desc="Processing review comments") as pbar:
            review_comment_polarity = []
            for review_comment in data_set['review_comment']:
                polarity = classifier(review_comment)[0]['score']
                review_comment_polarity.append(polarity)
                pbar.update(1)  # Update progress bar

            data_set['review_comment_polarity'] = review_comment_polarity

    except Exception as e:
        print(e)

    if data_set['product_id'].nunique() > 1:
        df['rtp_bert'] = data_set.groupby('product_id')['review_title_polarity'].mean()
        df['rcp_bert'] = data_set.groupby('product_id')['review_comment_polarity'].mean()
    return df


def calculate_average_sentiment(data_set):
    df = pd.DataFrame()

    df_textblob = calculate_sentiment_textblob(data_set)
    df_vader = calculate_sentiment_vader(data_set)
    df_bert = calculate_sentiment_bert(data_set)

    for i in range(len(df_textblob)):
        product_id = df_textblob.index[i]
        avg_pt = (df_textblob.iloc[i]['rtp_textblob'] + df_vader.iloc[i]['rtp_vader'] + df_bert.iloc[i][
            'rtp_bert']) / 3
        sentiment_st = df_textblob.iloc[i]['rts_textblob']
        avg_ct = (df_textblob.iloc[i]['rcp_textblob'] + df_vader.iloc[i]['rcp_vader'] + df_bert.iloc[i][
            'rcp_bert']) / 3
        sentiment_sc = df_textblob.iloc[i]['rcs_textblob']
        df.loc[i, 'product_id'] = product_id
        df.loc[i, 'avg_pt'] = avg_pt
        df.loc[i, 'sentiment_st'] = sentiment_st
        df.loc[i, 'avg_ct'] = avg_ct
        df.loc[i, 'sentiment_sc'] = sentiment_sc

    return df

# def database_recreate():
#     try:
#         conn = mc.connect(**sql_database_config)
#         cursor = conn.cursor()
#         cursor.execute("USE moody")
#         table_name = "moodyDB"
#         drop_query = "DROP TABLE IF EXISTS " + table_name
#         cursor.execute(drop_query)
#         create_query = ("CREATE TABLE IF NOT EXISTS moodyDB ("
#                         "id INT AUTO_INCREMENT PRIMARY KEY,product_id VARCHAR(255),"
#                         "product_id VARCHAR(255),"
#                         "avg_pt VARCHAR(255),"
#                         "sentiment_st VARCHAR(255),"
#                         "avg_ct VARCHAR(255),"
#                         "sentiment_sc VARCHAR(255))"
#                         )
#         cursor.execute(create_query)
#         conn.commit()
#         conn.close()
#     except mc.Error as e:
#         print(e)


def insert_data_into_database(data):
    try:
        conn = mc.connect(**sql_database_config)
        cursor = conn.cursor()
        cursor.execute("USE moody")
        insert_query = "INSERT INTO moodyDB (product_id,avg_pt,sentiment_st,avg_ct,sentiment_sc) VALUES (%s,%s,%s,%s,%s)"
        for i in data:
            cursor.execute(insert_query, i)
        conn.commit()
        conn.close()
    except mc.Error as e:
        print(e)


def main():
    df = create_data_set()
    df = data_cleaning(df)
    ne_df = calculate_average_sentiment(df)
    # df_textblob = calculate_sentiment_textblob(df)
    insert_data_into_database(ne_df.values.tolist())

    # collections = mongo_data_schema()
    # sample_data =[
    #     {"product_id": "1", "review_title": "title", "review_comment": "comment", "rating": 5},
    #     {"product_id": "2", "review_title": "title", "review_comment": "comment", "rating": 5},
    # ]
    #
    # collections.insert_many(sample_data)

    print("Hello World")


if __name__ == '__main__':
    main()
