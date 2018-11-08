"""
TODO: write method to read tweets from .txt to memory
"""


import tweepy
import time

consumer_key = ""
consumer_secret = ""

access_token = ""
access_secret = ""

"""
ID of first tweet written after 1/20/17 @ 12:00PM which is the official
time he became president
"""
first_tweet_id = 822507434396753921

"""
writes tweets to file. 
avoid using and stick to .txt file reading to not constantly query Twitter
which can limit our ability to do so.
"""
def getTweets():
    #setting up OAuthHandler and API. DO NOT CHANGE
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    api = tweepy.API(auth)

    #tweet handler
    cursor = tweepy.Cursor(api.user_timeline, screen_name = "realDonaldTrump", tweet_mode = "extended", since_id = first_tweet_id).items()
    
    fh = open("data/tweet_data.txt","w")
    
    while True:
        try: 
            tweet = cursor.next()
            #filters out retweets
            if not tweet.retweeted and "RT @" not in tweet.full_text:
                """
                NOTE: when importing the tweet text/content, there are 
                characters that python won't convert to because it only uses 
                the ASCII character set. The below line converts to utf-8
                to avoid this issue. DO NOT DELETE
                """
                tweet_text = tweet.full_text.encode("utf-8")
                #line to be written to file
                line = str(tweet.created_at) + "\t" + str(tweet.id)+ "\t" + str(tweet_text) + "\n"
                fh.writelines(line)
        
        #if timeout occurs, wait out the time out
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        
        #end loop condition
        except StopIteration:
            break
    
    fh.close()
