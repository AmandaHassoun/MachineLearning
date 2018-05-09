import sys
from sklearn.tree import export_graphviz
from sklearn.feature_selection import mutual_info_classif
import graphviz
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import operator
import math 
from decimal import *
import os

FAKE_TOTAL=1280
REAL_TOTAL=1280
VOCAB = set()
np.random.seed(100)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def fetch_data(filename):
    """Finds occurence of all words in titles and returns dict of key=word,
       value=occurence."""

    titles = []

    with open(filename) as fp:  
         line = fp.readline()
         while line:
             titles.append(line.strip())
             line = fp.readline() 

    return titles

def init_(train_size, train_set_f, train_set_r, vocab):
    """"""
    x = np.zeros((train_size*2, len(VOCAB)))
    y = np.zeros((1, train_size*2))

    for i in range(train_size):
       words = train_set_f[i].split()
       for j in range(len(words)):
           x[i][vocab.index(words[j])] = 1
           y[0][i] = 1 

    for i in range(train_size):
       words = train_set_r[i].split()
       for j in range(len(words)):
           x[i+train_size][vocab.index(words[j])] = 1
           y[0][i+train_size] = 0 

    return x,y
 
def count_occurrence(titles, start, end):
    """"""

    word_occurence = dict()
    set_list = titles

    for title in set_list:
        words = set(title.split())
        for word in words:
            VOCAB.add(word) 
            if word not in word_occurence:
                word_occurence[word] = 1
            else:
                word_occurence[word] += 1

    print(word_occurence["just"]) 
    print(sum(word_occurence.values()))
    return set_list[start:end]

def plot_performance(x, y_train, y_valid, y_test):
    """Plot learning curve."""

    plt.plot(x, y_train, label="Training Set Accuracy")
    plt.plot(x, y_valid, label="Validation Set Accuracy")
    plt.plot(x, y_test, label="Test Set Accuracy")
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('part7a.png', bbox_inches='tight')
    plt.show()

def accuracy(labels, y_test):
    length = y_test.shape[0]
    correct = 0 

    for i in range(length):
        if labels[i] == y_test[i]:
            correct += 1

    accur = Decimal(correct)/length
    print 'LR Accuracy: %.5f' % (accur*100)
    return accur*100

if __name__ == "__main__":
    
    fake_titles = fetch_data('clean_fake.txt')
    real_titles = fetch_data('clean_real.txt')

    #Set sizes 
    train_size = 900 
    valid_size = 190  
    test_size = 190

    #Populate vocabulary dict to get all words
    train_set_f = count_occurrence(fake_titles, 0, train_size)
    train_set_r = count_occurrence(real_titles, 0, train_size) 

    """
    sorted_vocab = sorted(VOCAB)
    list_sorted_vocab = list(sorted_vocab)

    #Preparing test set
    test_set_f = count_occurrence(fake_titles, train_size + valid_size, train_size + valid_size + test_size)
    test_set_r = count_occurrence(real_titles, train_size + valid_size, train_size + valid_size + test_size)
    x_test, y_test = init_(test_size, test_set_f, test_set_r, list_sorted_vocab)

    #Preparing validation set
    valid_set_f = count_occurrence(fake_titles, train_size, train_size + valid_size)
    valid_set_r = count_occurrence(real_titles, train_size, train_size + valid_size)
    x_valid, y_valid = init_(valid_size, valid_set_f, valid_set_r, list_sorted_vocab)

    m = 0
    for i in range(len(list_sorted_vocab)):
        if list_sorted_vocab[i] in ENGLISH_STOP_WORDS:
            m += 1
     
    x_train, y_train = init_(train_size, train_set_f, train_set_r, list_sorted_vocab)

    x = []
    y_t = []
    y_tst = []
    y_v = []

    #Part 7.a): plot performance of model on validation set vs. max_depth
    print("=================Running part 7==========================\n")
    print("Plotting performance of model vs. max_depth \n")

    best_accur = 0
    best_accur_test = 0
    best_accur_train = 0

    for i in range(10,200,20):
        x.append(i)
        clf = tree.DecisionTreeClassifier(max_depth=i, min_samples_split=20, max_features=500)
        clf = clf.fit(x_train, y_train.flatten())
        y_pred = clf.predict(x_valid)
        y_pred_t = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        accur_v = accuracy(y_pred, y_valid.flatten())
        accur_test = accuracy(y_pred_test, y_test.flatten())
        accur_t = accuracy(y_pred_t, y_train.flatten())
        y_t.append(accur_t)
        y_v.append(accur_v)
        y_tst.append(accur_test)
        if accur_v > best_accur:
            best_accur = accur_v
            best_accur_test = accur_test
            best_accur_train = accur_t
            max_d = i

    print 'Best max depth: %d \n' % (max_d)
    print("Final accuracy")
    print("Training set")
    print(best_accur_train)
    print("Validation set")
    print(best_accur)
    print("Test set")
    print(best_accur_test)
    print("\n")
    plot_performance(x,y_t,y_v,y_tst)
    """
    #Part 7.b) 
    """
    print("Generating image to visualize first 2 layers of the tree \n")
     
    clf = tree.DecisionTreeClassifier(max_depth=50, min_samples_split=20, max_features=500)
    clf = clf.fit(x_train, y_train.flatten())
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("part8-a") 
    
    print("Most important features from Decision Tree\n")
    print(list_sorted_vocab[5802])
    print(list_sorted_vocab[2826])
    print(list_sorted_vocab[5351])
    print(list_sorted_vocab[5351])
    print(list_sorted_vocab[4679])
    print(list_sorted_vocab[1718])
    """
