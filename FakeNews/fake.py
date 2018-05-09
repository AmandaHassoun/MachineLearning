import sys
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import operator
import math 
from decimal import *
import os

FAKE_TOTAL=1280
REAL_TOTAL=1280

def fetch_data(filename):
    """Returns list of lines from <filename> file.""" 

    titles = []

    with open(filename) as fp:  
         line = fp.readline()
         while line:
             titles.append(line.strip())
             line = fp.readline() 

    return titles

def count_occurrence(titles, start, end):
    """Finds occurence of all words in titles and returns 
       dict of key=word, value=count/total set size."""

    word_occurence = dict()
    set_list = titles[start:end] 

    for title in set_list:
        words = set(title.split())
        for word in words:
            if word not in word_occurence:
                word_occurence[word] = 1
            else:
                word_occurence[word] += 1
    
    return word_occurence, set_list

#prob(word|(fake or real news))
def cond_prob(word, word_count_fake, word_count_real,fake_total, real_total, m, p_hat):
    """Calculates the probability of a word being positive/negative given that it was seen
       in a fake or real news headline."""

    prob_f = math.log((word_count_fake.get(word, 0) + Decimal(m*p_hat))/Decimal(fake_total + m)) 
    prob_r = math.log((word_count_real.get(word, 0) + Decimal(m*p_hat))/Decimal(real_total + m))

    return prob_r, prob_f

def cond_prob_top10(word, word_count_fake, word_count_real,fake_total, real_total, m, p_hat):
    """Calculates the probability of a word being postive/negative given that it was seen
       in a fake or real news headline."""

    prob_f = (word_count_fake.get(word, 0) + Decimal(m*p_hat))/Decimal(fake_total + m)
    prob_r = (word_count_real.get(word, 0) + Decimal(m*p_hat))/Decimal(real_total + m)

    prob_f *= Decimal(0.5)
    prob_r *= Decimal(0.5)

    return prob_r, prob_f

def cond_prob_top10_absence(word, word_count_fake, word_count_real,fake_total, real_total, m, p_hat):
    """Calculates the probability of a word being postive/negative given that it was seen
       in a fake or real news headline."""

    prob_f = (len(word_count_fake.keys()) - (word_count_fake.get(word, 0) + Decimal(m*p_hat)))/Decimal(len(word_count_fake.keys()) + m)
    prob_r = (len(word_count_real.keys()) - (word_count_real.get(word, 0) + Decimal(m*p_hat)))/Decimal(len(word_count_real.keys()) + m)

    prob_f *= Decimal(0.5)
    prob_r *= Decimal(0.5)

    return prob_r, prob_f

def prob_class_given_title(title, word_count_fake, word_count_real, fake_total, real_total, m, p_hat):
    """Return p(fake|title) & p(real|title)."""

    prob_r = 0
    prob_f = 0 

    words = title.split()
    for word in words:
        temp1 , temp2 = cond_prob(word, word_count_fake, word_count_real, fake_total, real_total, m, p_hat) 
        prob_r += temp1
        prob_f += temp2

    return math.exp(prob_r) , math.exp(prob_f)

def predict(title, word_count_fake, word_count_real, fake_total, real_total, m, p_hat):
    """Returns whether given title is more likely to be a fake or real news headline."""

    real_news, fake_news = prob_class_given_title(title, word_count_fake, word_count_real, fake_total, real_total, m, p_hat)

    real_news *= 0.5 
    fake_news *= 0.5 

    if fake_news >= real_news:
        return 1 
    else:
        return 0

def best_mp(validation_set_f, validation_set_r, real_total, fake_total):
    """Returns m & p that maximize accuracy on the validation set."""

    fake_f = 0
    fake_r = 0
    real_f = 0
    real_r = 0
    best_perf = 0
    p_range = xfrange(0.001, 0.5, 0.05) 
    m_range = xfrange(0.1, 2, 0.05) 

    for p in p_range: 
        for m in m_range:
            for title in validation_set_f:
               result = predict(title, occur_training_fake, occur_training_real, fake_total, real_total, m,p)
               if result == 1:
                   fake_f += 1
               else:
                   fake_r += 1

            for title in validation_set_r:
               result = predict(title, occur_training_fake, occur_training_real, real_total, fake_total, m,p)
               if result == 1:
                   real_f += 1
               else:
                   real_r += 1

            if (100*Decimal(fake_f + real_r)/(real_total + fake_total)) > best_perf:
                best_perf = 100*Decimal(fake_f + real_r)/(len(validation_set_f) + len(validation_set_r))
                print 'perf %.5f ' % (best_perf)
                print 'r %d ' % (real_r)
                print 'f %d ' % (fake_f)
                m_ = m
                p_ = p

            fake_f = 0
            fake_r = 0
            real_f = 0
            real_r = 0

    print 'BEST PERF %.5f ' % (best_perf)
    print 'm : %.5f ' % (m_)
    print 'p : %.5f ' % (p_)

    return m_, p_

def get_perf(real_set, fake_set, real_total, fake_total, m, p):
    """Returns accuracy of NB."""

    fake_f = 0
    fake_r = 0
    real_f = 0
    real_r = 0

    for title in real_set:
        result = predict(title, occur_training_fake, occur_training_real, real_total, fake_total, m,p)
        if result == 1:
            fake_f += 1
        else:
            fake_r += 1

    for title in fake_set:
        result = predict(title, occur_training_fake, occur_training_real, real_total, fake_total, m,p)
        if result == 1:
            real_f += 1
        else:
            real_r += 1

    best_perf = 100*Decimal(fake_f + real_r)/(real_total + fake_total)
    print(real_total + fake_total)
    print 'perf %.5f ' % (best_perf)
    print 'r %d ' % (real_r)
    print 'f %d ' % (fake_f)

def top10(training_set, word_count_fake, word_count_real, real_total, fake_total, m, p_hat):
    """"""
    _in_fake = dict()
    _in_real = dict()

    abs_r = dict()
    abs_f = dict() 

    for word in training_set:
        prob_real , prob_fake = cond_prob_top10(word, word_count_fake, word_count_real, real_total, fake_total, m, p_hat)
        _in_fake[word] = prob_fake
        _in_real[word] = prob_real

    
    for key, val in training_set.iteritems():
        prob_real , prob_fake = cond_prob_top10_absence(key, word_count_fake, word_count_real, real_total, fake_total, m, p_hat) 
        abs_r[key] = prob_real
        abs_f[key] = prob_fake 
     
    absense_real = dict(sorted(abs_r.items(), key=operator.itemgetter(1), reverse=True)[:10])
    absense_fake = dict(sorted(abs_f.items(), key=operator.itemgetter(1), reverse=True)[:10])
    presence_real = dict(sorted(_in_real.items(), key=operator.itemgetter(1), reverse=True)[:10])
    presence_fake = dict(sorted(_in_fake.items(), key=operator.itemgetter(1), reverse=True)[:10])

    print("Presence real \n")
    for word in presence_real:
        print(word)
    print("Presence fake \n")
    for word in presence_fake:
        print(word)
    print("Absense real \n")
    for word in absense_real:
        print(word)
    print("Absense fake \n")
    for word in absense_fake:
        print(word)

def top10_non_stopwords(training_set, word_count_fake, word_count_real, real_total, fake_total, m, p_hat):
    """"""
    _in_fake = dict()
    _in_real = dict()

    for word in training_set:
        if word not in ENGLISH_STOP_WORDS:
            prob_real , prob_fake = cond_prob_top10(word, word_count_fake, word_count_real, real_total, fake_total, m, p_hat)
            _in_fake[word] = prob_fake
            _in_real[word] = prob_real

    presence_real = dict(sorted(_in_real.items(), key=operator.itemgetter(1), reverse=True)[:10])
    presence_fake = dict(sorted(_in_fake.items(), key=operator.itemgetter(1), reverse=True)[:10])

    print("Presence real \n")
    for word in presence_real:
        print(word)
    print("Presence fake \n")
    for word in presence_fake:
        print(word)

def xfrange(start, stop, step):
    i = 0
    ran = []
    while start + i * step < stop:
        ran.append(start + i * step)
        i += 1

    return ran
 
if __name__ == "__main__":
    fake_titles = fetch_data('clean_fake.txt')
    real_titles = fetch_data('clean_real.txt')

    #Set sizes 
    train_set = 900 
    valid_set = 190  
    test_set = 190

    occur_training_fake, training_set_f = count_occurrence(fake_titles, 0, train_set)
    occur_validation_fake, validation_set_f = count_occurrence(fake_titles, train_set, train_set + valid_set)
    occur_test_fake, test_set_f = count_occurrence(fake_titles, train_set + valid_set, train_set + valid_set + test_set)

    occur_training_real, training_set_r = count_occurrence(real_titles, 0, train_set)
    occur_validation_real, validation_set_r = count_occurrence(real_titles, train_set, train_set + valid_set)
    occur_test_real, test_set_r = count_occurrence(real_titles, valid_set + train_set, train_set + valid_set + test_set) 

    real_total = len(occur_validation_real.keys())
    fake_total = len(occur_validation_fake.keys())

    #Part 2: tuning m&p using validation set
    print("==============Running part 2===================\n")
    #Using validation set to find the best m and p_hat
    #m, p = best_mp(validation_set_f, validation_set_r, real_total, fake_total)
    #Best m & p
    
    m = 1.85 
    p = 0.251

    test_total_f = len(occur_test_fake.keys()) 
    valid_total_f = len(occur_validation_fake.keys()) 
    training_total_f = len(occur_training_fake.keys()) 

    test_total_r = len(occur_test_real.keys())
    valid_total_r = len(occur_validation_real.keys())
    training_total_r = len(occur_training_real.keys())

    print("Performance of Naive Bayes on all 3 sets using best m&p \n")
    print("Test set performance")
    get_perf(test_set_f, test_set_r,test_set, test_set, m, p)
    print("Validation set performance")
    get_perf(validation_set_f, validation_set_r, valid_set, valid_set, m, p)
    print("Training set performance")
    get_perf(training_set_f, training_set_r,train_set, train_set, m, p)

    print("=================Running part 3===================\n")
    print("Top10\n")
    #Part 3.a)
    training_set = Counter(occur_training_fake) + Counter(occur_training_real)
    top10(training_set, occur_training_fake, occur_training_real, train_set, train_set, m, p)
    #Part 3.b)
    print("Top10 - EXCLUDING STOP WORDS\n")
    top10_non_stopwords(training_set, occur_training_fake, occur_training_real, train_set, train_set, m, p) 
