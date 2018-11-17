import os
import csv
import math
from collections import defaultdict, Counter

POS_LABEL = 'positive'
NEG_LABEL = 'negative'
NEUT_LABEL = 'neutral'

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, path_to_data, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        # self.train_dir = os.path.join(path_to_data, "train")
        # self.test_dir = os.path.join(path_to_data, "test")
        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0,
                                        NEUT_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0,
                                         NEUT_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float),
                                   NEUT_LABEL: defaultdict(float)}

    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """
        with open("tweets2.txt") as tsv:
            for tweet in csv.reader(tsv, dialect="excel-tab"):
                self.tokenize_and_update_model(tweet[2], tweet[1])
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEUT_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEUT_LABEL])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        for word in bow:
            self.vocab.add(word)
            self.class_word_counts[label][word] += bow[word]
            self.class_total_word_counts[label] += bow[word]

        self.class_total_doc_counts[label] += 1

    def tokenize_and_update_model(self, doc, label):
        """
        Implement me!

        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """
        self.update_model(self.tokenize_doc(doc), label)

    def top_n(self, label, n):
        """
        Implement me!
        
        Returns the most frequent n tokens for documents with class 'label'.
        """
        top = []
        count = 0
        for frequency in sorted(self.class_word_counts[label].values(), reverse=True):
            if count > n:
                break
            top.append((list(self.class_word_counts[label].keys())[list(self.class_word_counts[label].values()).index(frequency)], frequency))
            count += 1
        return top


    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label
        according to this NB model.
        """
        total_count = self.class_total_word_counts[label]
        return ((self.class_word_counts[label][word] + 0.00001)/(total_count+ 0.00001 * len(self.vocab)))

    # def p_word_given_label_and_alpha(self, word, label, alpha):
    #     """
    #     Implement me!

    #     Returns the probability of word given label wrt psuedo counts.
    #     alpha - smoothing parameter
    #     """
    #     count_in_class = self.class_total_word_counts[label]
    #     return ((self.class_word_counts[label][word] + alpha)/(count_in_class+ alpha * len(self.vocab)))
        

    def log_likelihood(self, bow, label):
        """
        Implement me!

        Computes the log likelihood of a set of words given a label and smoothing.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; smoothing parameter
        """
        total = 0
        count = 0
        for word in bow:
            if self.p_word_given_label(word, label) != 0:
                total += math.log(self.p_word_given_label(word, label), 2)
        return total

    def log_prior(self, label):
        """
        Implement me!

        Returns the log prior of a document having the class 'label'.
        """
        return math.log(self.class_total_doc_counts[label]/(self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL]+ self.class_total_doc_counts[NEUT_LABEL]), 2)
        
    def unnormalized_log_posterior(self, bow, label):
        """
        Implement me!

        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return (self.log_prior(label) + self.log_likelihood(bow, label))

    def classify(self, bow):
        """
        Implement me!

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """
        pos = self.unnormalized_log_posterior(bow, POS_LABEL)
        neg = self.unnormalized_log_posterior(bow, NEG_LABEL)
        neut = self.unnormalized_log_posterior(bow, NEUT_LABEL)
        
        if pos > neg:
            if pos > neut:
                return POS_LABEL
            else:
                return NEUT_LABEL
        else:
            if neg > neut:
                return NEG_LABEL
            else:
                return NEUT_LABEL

    def likelihood_ratio(self, word):
        """
        Implement me!

        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return (self.p_word_given_label(word, POS_LABEL)/self.p_word_given_label(word, NEG_LABEL))/self.p_word_given_label(word, NEUT_LABEL)

    def evaluate_classifier_accuracy(self):
        """
        DO NOT MODIFY THIS FUNCTION

        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0

        with open("tweets2.txt") as tsv:
            for tweet in csv.reader(tsv, dialect="excel-tab"):
                self.tokenize_and_update_model(tweet[2], tweet[1])
                content = tweet[2]
                bow = self.tokenize_doc(content)
                label = tweet[1]
                if self.classify(bow) == label:
                    correct += 1.0
                total += 1.0
        return 100 * correct / total

if __name__ == "__main__":
    nb = NaiveBayes("tweets1.txt", tokenizer=tokenize_doc)
    nb.train_model()
    print(nb.evaluate_classifier_accuracy())

    neg = 0
    pos = 0
    neut = 0
    lines = [line.rstrip('\n') for line in open('trump_tweets.txt')]
    for line in lines:
        label = nb.classify(line)
        if label == "negative":
            neg += 1
        elif label == "positive":
            pos += 1
        else:
            neut += 1
    print(pos, neg, neut)
    # print(nb.classify("Isn’t it ironic that large Caravans of people are marching to our border wanting U.S.A. asylum because they are fearful of being in their country - yet they are proudly waving"))

