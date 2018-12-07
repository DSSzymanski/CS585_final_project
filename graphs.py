# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from model_and_rating_data import get_class, get_ratings

def get_data():
    ratings = get_ratings()
    tweet_pred = get_class()
    data = []
    for item in ratings:
        data.append([item[0],[0,0,0],item[1]])
    counter = 0
    for tweet in tweet_pred:
        split = tweet.split()
        if split[0]>data[counter][0]:
            counter += 1
        data[counter][1][int(split[1])] += 1
    return data

def rating_vs_tweetclass(r,t,data):
    """
    r = rating type
    0 = approval
    1 = disapproval
    2 = no opinion
    
    t = tweet type
    0 = pos
    1 = neg
    2 = neu
    """
    rating_label = ""
    if r == 0: rating_label = "Approve"
    elif r == 1: rating_label = "Disapprove"
    else: rating_label = "No Opinion"
    
    tweet_label = ""
    if t == 0: tweet_label = "Positive"
    elif t == 1: tweet_label = "Negative"
    else: tweet_label = "Neutral"
    
    
    date_axis = []
    tweets = []
    rating = []
    for entry in data:
        date_axis.append(entry[0])
        tweets.append(int(entry[1][t]))
        rating.append(int(entry[2][r]))
    
    #graph
    plt.figure(num=1, figsize=(16,16))
    #top
    plt.subplot(211)
    plt.xlabel("Time")
    plt.ylabel("Num " + tweet_label + " tweets")
    plt.xticks([])
    plt.plot(date_axis,tweets)
    #bottom
    plt.subplot(212)
    plt.xlabel("Time")
    plt.ylabel("% " + rating_label)
    plt.xticks([])
    plt.plot(date_axis,rating)
    plt.show()
    
def main():
    data = get_data()
    for x in range(3):
        for y in range(3):
            rating_vs_tweetclass(x,y,data)
            
        
    
if __name__ == "__main__":
    main()