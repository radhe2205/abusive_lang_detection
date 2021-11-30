import tweepy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from collections import Counter

def gen_api():
    auths = {
        'consumer':None,
        'consumer_secret':None,
        'access':None,
        'access_secret':None
    }
    with open('tokens.txt') as f:
        tokens = f.readlines()
        for token,auth in zip(tokens,auths.keys()):
            token = token.strip('\n')
            auths[auth] = token

    auth = tweepy.OAuthHandler(auths['consumer'],auths['consumer_secret'])
    auth.set_access_token(auths['access'],auths['access_secret'])
    api = tweepy.API(auth)
    return api

def clean_tweet(tweet):
    tweet = re.sub('@[A-Za-z0-9]+','@USER',tweet)
    tweet = re.sub('http\S+', 'URL', tweet)
    return tweet

def collect_tweet(id,api):
    try:
        tweet = api.get_status(id)
        tweet = clean_tweet(tweet.text)
        return tweet
    except:
        return None

def sample_data(data, n=250000):
    negatives = data[data['label']=='NOT'].sample(n=n//2)
    positives = data[data['label']=='OFF'].sample(n=n//2)
    data = positives.append(negatives, ignore_index=True)
    return data

def clean_tweet_dataset(data):
    data = data[data['id'].notna()]
    data.columns = ['tweet','label']
    return data

if __name__ == "__main__":
    # data = pd.read_csv('data/SOLID/task_a_distant_clean.tsv', sep='\t', index_col=False)
    # data = sample_data(data)
    # api = gen_api()
    # data['id'] = data['id'].apply(collect_tweet, api=api)
    # data.to_csv('data/SOLID/task_a_distant_tweets.tsv', sep='\t', index=False)
    data = pd.read_csv('data/SOLID/task_a_distant_tweets_clean.tsv', sep='\t', lineterminator='\n')
    data.to_csv('data/SOLID/task_a_distant_tweets.tsv', sep='\t', index=False)
