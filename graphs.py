# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
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
    
def all_vs_rating(r,data):
    """
    r = rating type
    0 = approval
    1 = disapproval
    2 = no opinion
    """
    rating_label = ""
    if r == 0: rating_label = "Approve"
    elif r == 1: rating_label = "Disapprove"
    else: rating_label = "No Opinion"
    
    date_axis = []
    tweets = []
    rating = []
    for entry in data:
        date_axis.append(entry[0])
        tweets.append(sum(entry[1]))
        rating.append(int(entry[2][r]))
    
    #graph
    plt.figure(num=1, figsize=(16,16))
    #top
    plt.subplot(211)
    plt.xlabel("Time")
    plt.ylabel("Num " + "all" + " tweets")
    plt.xticks([])
    plt.plot(date_axis,tweets)
    #bottom
    plt.subplot(212)
    plt.xlabel("Time")
    plt.ylabel("% " + rating_label)
    plt.xticks([])
    plt.plot(date_axis,rating)
    plt.show()
    
def get_correlations(data):
    approval = []
    disapproval = []
    no_opinion = []
    
    positive = []
    negative = []
    neutral = []
    
    week_axis = []
    count = 0
    for entry in data:
        week_axis.append(count)
        approval.append(int(entry[2][0]))
        disapproval.append(int(entry[2][1]))
        no_opinion.append(int(entry[2][2]))
        positive.append(entry[1][0])
        negative.append(entry[1][1])
        neutral.append(entry[1][2])
        count += 1
    
    print("negative and approval\t" + str(pearsonr(negative, approval)[0]))
    print("negative and disapproval\t" + str(pearsonr(negative, disapproval)[0]))
    print("negative and no opinion\t" + str(pearsonr(negative, no_opinion)[0]))
    print()
    print("positive and approval\t" + str(pearsonr(positive, approval)[0]))
    print("positive and disapproval\t" + str(pearsonr(positive, disapproval)[0]))
    print("positive and no opinion\t" + str(pearsonr(positive, no_opinion)[0]))
    print()
    print("neutral and approval\t" + str(pearsonr(neutral, approval)[0]))
    print("neutral and disapproval\t" + str(pearsonr(neutral, disapproval)[0]))
    print("neutral and no opinion\t" + str(pearsonr(neutral, no_opinion)[0]))
    
        
def main():
    get_correlations(get_data())
    
    """
    data = get_data()
    tweets = [0,0,0]
    for x in range(len(data)):
        tweets[0] += data[x][1][0]
        tweets[1] += data[x][1][1]
        tweets[2] += data[x][1][2]
    print(tweets, sum(tweets))
    
    for x in range(3):
        all_vs_rating(x,data)
    """        
        
    
if __name__ == "__main__":
    main()