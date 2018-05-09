import sys
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
    """Returns x and y: y is the expected output (0 if headline is real and
       1 if fake, and x is the dictionnary of unique words found in the
       training set)."""

    x = np.zeros((len(VOCAB), train_size*2))
    y = np.zeros((1, train_size*2))

    for i in range(train_size):
       words = train_set_f[i].split()
       for j in range(len(words)):
           x[vocab.index(words[j])][i] = 1
           y[0][i] = 1 

    for i in range(train_size):
       words = train_set_r[i].split()
       for j in range(len(words)):
           x[vocab.index(words[j])][i+train_size] = 1
           y[0][i+train_size] = 0 

    return x,y
 
def count_occurrence(titles, start, end):
    """Returns list of headlines for all sets. Also populates VOCAB
       set."""

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
    
    return set_list[start:end]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def f(features, theta):   
    return sigmoid(np.dot(theta.T, features))

def cost(features,y,theta):
    num_observations = y.shape[1] 
    h_theta = f(features, theta)
    J = 1/float(num_observations) * np.sum(-y * np.log(h_theta) - (1-y) * np.log(1-h_theta))
    return J

def gradient_descent(features,y,theta,alpha, lambda_):
    prediction = f(features, theta)
    gradient = np.dot(features, (prediction - y).T) + ((lambda_/y.shape[1])*theta) 
    gradient /= features.shape[0] 
    gradient *= alpha
    theta -= gradient 
    return theta

def classify(preds):
  """Decision boundary classification."""

  result = np.empty_like(preds)
  for i in range(preds.shape[1]):
      if preds[0][i] >= 0.5:
          result[0][i] = 1
      else:
          result[0][i] = 0 

  return result 

def accuracy(theta, x_test, y_test):
    correct = 0
    length = y_test.shape[1]
    pred = f(x_test, theta)
    preds = classify(pred)
    correct = 0 

    for i in range(y_test.shape[1]):
        if preds[0][i] == y_test[0][i]:
            correct += 1

    accur = Decimal(correct)/length
    print 'LR Accuracy: %.5f' % (accur*100)
    return accur

def train(features,y,alpha,theta,num_iters, x_test, y_test, x_valid, y_valid, lambda_):

    perf_train = []
    perf_test = []
    perf_valid = []
    iters = []

    for x in range(num_iters):
        new_theta = gradient_descent(features, y, theta, alpha, lambda_)
        theta = new_theta
        if x % 100 == 0:
            print('cost: ', cost(features,y,theta))
            print('Validation set')
            valid_accur = accuracy(theta, x_valid, y_valid)
            print('Test set')
            test_accur = accuracy(theta, x_test, y_test)
            print('Training set')
            train_accur = accuracy(theta, x_train, y_train)
            perf_train.append(train_accur)
            perf_valid.append(valid_accur)
            perf_test.append(test_accur)
            iters.append(x)

    return theta, perf_train, perf_valid, perf_test, iters

def best_params(x_train,y_train,alpha,initial_theta,iterations, x_valid, y_valid):
    """Find best accuracy using the validation set by tweeking lambda and
       alpha parameters."""

    best_perf = 0
    best_lambda = 0
    best_alpha = 0

    for lambda_ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
       for alpha in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]: 
           for x in range(iterations):
               new_theta = gradient_descent(x_train, y_train, initial_theta, alpha, lambda_)
               initial_theta = new_theta
               if x % 100 == 0:
                   print('cost: ', cost(x_train,y_train,initial_theta))
                   print('Validation set')
                   perf_valid = accuracy(initial_theta, x_valid, y_valid)
                   print('Training set')
                   perf_train = accuracy(initial_theta, x_train, y_train)

           if perf_valid > best_perf:
               best_perf = perf_valid
               best_lambda = lambda_
               best_alpha = alpha

    print("Best Params (using validation set)\n")
    print 'Accur %.5f' % (best_perf*100)
    print 'Lambda %.5f' % (best_lambda)
    print 'Alpha %.5f' % (best_alpha)

    return best_lambda, best_alpha

def plot_performance(x, y_train, y_test, y_valid):
    """Plot learning curve."""

    plt.plot(x, y_train, label="Training Set Accuracy")
    plt.plot(x, y_test, label="Test Set Accuracy")
    plt.plot(x, y_valid, label="Validation Set Accuracy")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('part4-learning.png', bbox_inches='tight')
    plt.show()

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

    sorted_vocab = sorted(VOCAB)
    list_sorted_vocab = list(sorted_vocab)

    x_train, y_train = init_(train_size, train_set_f, train_set_r, list_sorted_vocab)

    valid_set_f = count_occurrence(fake_titles, train_size, train_size + valid_size)
    valid_set_r = count_occurrence(real_titles, train_size, train_size + valid_size)

    test_set_f = count_occurrence(fake_titles, train_size + valid_size, train_size + valid_size + test_size)
    test_set_r = count_occurrence(real_titles, train_size + valid_size, train_size + valid_size + test_size)

    x_test, y_test = init_(test_size, test_set_f, test_set_r, list_sorted_vocab)
    x_valid, y_valid = init_(valid_size, valid_set_f, valid_set_r, list_sorted_vocab)
    initial_theta = np.zeros((x_train.shape[0],1))

    alpha = 0.4 
    iterations = 2000 
    lambda_ = 0.2

    #Part 4 : Find best combo of (alpha,lambda)
    print("==============Running part 4=====================\n") 
    print("Trying to find best lambda&alpha \n")
    #lambda_, alpha = best_params(x_train,y_train,alpha,initial_theta,iterations, x_valid, y_valid)

    print("Training model \n")
    #Train model using best combo found
    final_theta, perf_train, perf_valid, perf_test, iters = train(x_train,y_train,alpha,initial_theta,iterations,x_test,y_test, x_valid, y_valid, lambda_)

    print("\n")
    print("FINAL Performance")
    print("Test set")
    accuracy(final_theta, x_test, y_test)
    print("Validation set")
    accuracy(final_theta, x_valid, y_valid)
    print("Training set")
    accuracy(final_theta, x_train, y_train)

    print("\n")
    #plot_performance(iters, perf_train, perf_test, perf_valid)

    sort = np.argsort(final_theta, axis=0)
    top_10_neg = sort[-10:]
    top_10_pos = sort[1:11:1]

    #Part 6.a)
    print("===============Running part6====================\n")
    print("TOP10 \n")
    print("Top10 negative thetas")
    for i in range(len(top_10_pos)):
        print(list_sorted_vocab[top_10_pos[i][0]])
        print(final_theta[top_10_pos[i][0]][0])

    print("\n")
 
    print("Top10 positive thetas")
    for i in range(len(top_10_neg)):
        print(list_sorted_vocab[top_10_neg[i][0]])
        print(final_theta[top_10_neg[i][0]][0])

    print("\n")

    #Part 6.b)
    print("TOP10 - EXCLUDING STOP WORDS \n")
    rem_stopwords = dict()
    for i in range(len(list_sorted_vocab)):
        if list_sorted_vocab[i] not in ENGLISH_STOP_WORDS:
            rem_stopwords[list_sorted_vocab[i]] = final_theta[i][0] 

    no_stop = sorted(rem_stopwords.items(), key=operator.itemgetter(1), reverse=True) 
    top_10_neg_no_stop = no_stop[:10]
    top_10_pos_no_stop = no_stop[-1:-11:-1]

    print("Top10 negative thetas (no stop words)")
    for i in range(10):
        print(top_10_pos_no_stop[i][0])
        print(top_10_pos_no_stop[i][1])
    print("\n")
    print("Top10 positive thetas (no stop words)")
    for i in range(10): 
        print(top_10_neg_no_stop[i][0])
        print(top_10_neg_no_stop[i][1])
