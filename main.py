import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import mysql.connector
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.special import softmax


# connect to the mongoDB
def connect_mongo():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['test']
    collection = db['test_collection']
    return collection


def send_data_to_mongo(df):
    collection = connect_mongo()
    collection.insert_many(df.to_dict('records'))


# connect to the mysql
def connect_mysql():
    db = mysql.connector.connect(
        host="localhost",
        user="myusername",
        password="mypassword"
    )
    return db


# roberta model for sentiment analysis
def roberta_model():
    task = 'sentiment-analysis'
    pre_trained_model_name = f'cardiffnlp/twitter-roberta-base-{task}'
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name)
    return tokenizer, model


# sentiment analysis using roberta model
def roberta_sentiment(text):
    tokenizer, model = roberta_model()
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_val = {'r_neg': scores[0], 'r_neu': scores[1], 'r_pos': scores[2]}
    return scores_val


# sentiment analysis using textblob model
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity


def sentiment_analysis(df):
    textblob_polarity = []
    roberta_polarity = {}
    for i in tqdm(df.iterrows(), total=len(df), desc='TextBlob'):
        textblob_polarity.append(textblob_sentiment(df['text'][i]))

    for i in tqdm(df.iterrows(), total=len(df), desc='Roberta'):
        try:
            index = df['text'][i]
            roberta_polarity[index] = roberta_sentiment(df['text'][i])
        except RuntimeError:
            print('Broke at ', i)

    return textblob_polarity, roberta_polarity


def add_data_to_df(df, textblob_polarity, roberta_polarity):
    df['textblob_polarity'] = textblob_polarity
    df['roberta_polarity'] = roberta_polarity
    return df


def merge_data(df, result):
    df = pd.DataFrame(result).T
    df = df.reset_index().rename(columns={'index': 'text'})
    df = df.merge(df, how='left', on='text')
    return df


def main():
    plt.style.use('ggplot')
    # read the data from csv
    df = pd.read_csv('data/apple_reviews.csv')

    print(df.dtypes)

    # # connect to the mongoDB
    # collection = connect_mongo()
    # # insert the data into mongoDB
    # collection.insert_many(df.to_dict('records'))


if __name__ == '__main__':
    main()
