import pandas as pd
import pytest
from main import (data_cleaning,
                  calculate_sentiment_textblob,
                  calculate_sentiment_vader,
                  calculate_sentiment_bert,
                  calculate_average_sentiment)


@pytest.fixture
def sample_data():
    data = {
        'id': [1, 2, 3],
        'product_id': [1, 2, 1],
        'review_title': ['Good product', 'Not satisfied', 'Great experience'],
        'review_comment': ['Nice item', 'Poor quality', 'Excellent service'],
        'rating': [4, 2, 5]
    }
    return pd.DataFrame(data)


def test_data_cleaning(sample_data):
    cleaned_data = data_cleaning(sample_data)
    assert len(cleaned_data) == 3


def test_calculate_sentiment_textblob(sample_data):
    sentiment_data = calculate_sentiment_textblob(sample_data)
    assert len(sentiment_data) == 2
    assert sentiment_data['rtp_textblob'].iloc[0] == 0.75
    assert sentiment_data['rts_textblob'].iloc[0] == 0.675
    assert sentiment_data['rcp_textblob'].iloc[0] == 0.8
    assert sentiment_data['rcs_textblob'].iloc[0] == 1.0
    assert sentiment_data['rtp_textblob'].iloc[1] == -0.25
    assert sentiment_data['rts_textblob'].iloc[1] == 1.0
    assert sentiment_data['rcp_textblob'].iloc[1] == -0.4
    assert sentiment_data['rcs_textblob'].iloc[1] == 0.6


def test_calculate_sentiment_vader(sample_data):
    sentiment_data = calculate_sentiment_vader(sample_data)
    assert len(sentiment_data) == 2
    assert 'rtp_vader' in sentiment_data.columns
    assert 'rcp_vader' in sentiment_data.columns


def test_calculate_sentiment_bert(sample_data):
    sentiment_data = calculate_sentiment_bert(sample_data)
    assert len(sentiment_data) == 2
    assert 'rtp_bert' in sentiment_data.columns
    assert 'rcp_bert' in sentiment_data.columns


def test_calculate_average_sentiment(sample_data):
    avg_sentiment_data = calculate_average_sentiment(sample_data)
    assert len(avg_sentiment_data) == 2
    assert 'avg_pt' in avg_sentiment_data.columns
    assert 'sentiment_st' in avg_sentiment_data.columns
    assert 'avg_ct' in avg_sentiment_data.columns
    assert 'sentiment_sc' in avg_sentiment_data.columns
