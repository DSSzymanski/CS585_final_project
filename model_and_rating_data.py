import os
import numpy as np
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scraper import getRatingData
from src.utils import *
from src.date_dataset import *
from src.char_level_cnn import CharLevelConvNet

device = torch.device("cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alphabet", type=str,
                        default="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
    parser.add_argument("-m", "--max_length", type=int, default=285)
    parser.add_argument("-i", "--input", type=str, default="data", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    args = parser.parse_args()

    return args

def load_model():
    model = torch.load("output/char_level_trained_model")
    return model

#not used
"""
def load_old_tweets():
    fh = open("trump_tweets_2.txt", "r")
    getList = fh.readlines()
    fh.close()
    return getList
"""
 
def rating_date():
    ratings = getRatingData()
    ret_ratings = []
    for rating in ratings:
        date = rating[0].split('-')
        year = date[0][0:4]
        if len(date[1]) > 2:
            end_date = date[1].split()
            get_string = end_date[1]+'-'+end_date[0]+'-'+year
            new_string = datetime.datetime.strptime(get_string,'%d-%b-%Y').strftime('%Y%m%d')
        else:
            date = rating[0].split()
            get_string = date[2].split('-')[1] + '-' + date[1] + '-' + date[0]
            new_string = datetime.datetime.strptime(get_string,'%d-%b-%Y').strftime('%Y%m%d')
        ret_ratings.append((new_string, rating[1]))
    return ret_ratings

#ignore, writes file with new date format
"""
def tweet_redate():
    tweets = load_old_tweets()
    ret_list = []
    for tweet in tweets:
        get_string = tweet[8:10]+'-'+tweet[4:7]+'-'+tweet[26:30]
        new_string = datetime.datetime.strptime(get_string,'%d-%b-%Y').strftime('%Y%m%d')
        text = tweet[31:]
        ret_list.append(str(new_string + " " + text))
    fh = open("data/new_date_tweets_data.txt", "w")    
    fh.writelines(ret_list)
    fh.close()
"""

#not used
"""
def load_tweets():
    fh = open("data/new_date_tweets_data.txt", "r")
    getList = fh.readlines()
    fh.close()
    return getList
"""

def get_predictions(args):
    prediction_set = CharDataset(os.path.join(args.input, "new_date_tweets_data.txt"),
                           os.path.join(args.input, "SemEval/classes.txt"), args.alphabet, args.max_length)
    prediction_generator = DataLoader(prediction_set, shuffle=False)
    
    model = load_model()
    model = model.to(device)
    model.eval()
    
    pred_date = []
    pred_prob = []
    for batch in prediction_generator:
        _, n_date = batch         # numpy format
        n_date = n_date[0]

        #batch = [record.to(device) for record in batch]
        p_data, _ = batch               # tensor format

        p_predicted_prob = model(p_data)
        p_predicted_prob = F.softmax(p_predicted_prob, dim=1)
        pred_prob.append(p_predicted_prob)
        pred_date.append(n_date)
        
    pred_prob = torch.cat(pred_prob, 0).cpu().data.numpy()
    
    date_class = []
    for x in range(len(pred_date)):
        date_class.append(pred_date[x] + " " + str(np.argmax(pred_prob[x]).item()) + "\n")

    fh = open("data/date_class.txt", "w")    
    fh.writelines(date_class)
    fh.close()    

    

def main():
    args = get_args()
    get_predictions(args) 
    
if __name__ == "__main__":
    main()