#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

if __name__ == "__main__":
    input_path = os.path.join('..', 'data', 'SemEval', 'tweets2.txt')
    positive_tweets = []
    negative_tweets = []
    neutral_tweets = []
    with open(input_path, 'r') as infile:
        for line in infile:
            words = line.split()
            label_text = words[1]
            tweet = " ".join(words[2:]).lower()

            if label_text == 'positive':
                positive_tweets.append(tweet)
            elif label_text == 'negative':
                negative_tweets.append(tweet)
            elif label_text == 'neutral':
                neutral_tweets.append(tweet)
            else:
                print("Found invalid label: {}".format(label_text))

    min_num = min([len(positive_tweets), len(negative_tweets), len(neutral_tweets)])

    threshold = 0.8
    split_pos = int(threshold * min_num)
    # pos_split = int(threshold * len(positive_tweets))
    # neg_split = int(threshold * len(negative_tweets))
    # neu_split = int(threshold * len(neutral_tweets))

    # train_pos = positive_tweets[:pos_split]
    # train_neg = negative_tweets[:neg_split]
    # train_neu = neutral_tweets[:neu_split]
    # test_pos = positive_tweets[pos_split:]
    # test_neg = negative_tweets[neg_split:]
    # test_neu = neutral_tweets[neu_split:]
    train_pos = positive_tweets[:split_pos]
    train_neg = negative_tweets[:split_pos]
    train_neu = neutral_tweets[:split_pos]
    test_pos = positive_tweets[split_pos:min_num]
    test_neg = negative_tweets[split_pos:min_num]
    test_neu = neutral_tweets[split_pos:min_num]
    print(len(train_pos), len(test_pos))
    print(len(train_neg), len(test_neg))
    print(len(train_neu), len(test_neu))

    train_path = os.path.join('..', 'data', 'SemEval', 'Balanced', 'train.txt')
    test_path = os.path.join('..', 'data', 'SemEval', 'Balanced', 'test.txt')
    with open(train_path, 'w') as train_file, open(test_path, 'w') as test_file:
        for tweet in train_pos:
            train_file.write("0 " + tweet + '\n')
        for tweet in train_neg:
            train_file.write("1 " + tweet + '\n')
        for tweet in train_neu:
            train_file.write("2 " + tweet + '\n')
        for tweet in test_pos:
            test_file.write("0 " + tweet + '\n')
        for tweet in test_neg:
            test_file.write("1 " + tweet + '\n')
        for tweet in test_neu:
            test_file.write("2 " + tweet + '\n')
