import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import nltk
import transformers
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# connect to the mongoDB
def connect_mongo():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['test']
    collection = db['test_collection']
    return collection


def main():
    # read the data from csv
    df = pd.read_csv('data.csv')
    # connect to the mongoDB
    collection = connect_mongo()
    # insert the data into mongoDB
    collection.insert_many(df.to_dict('records'))


if __name__ == '__main__':
    main()
