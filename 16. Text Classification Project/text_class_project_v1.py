
# coding: utf-8

# ## Text Classification Using Naive Bayes
# ### Your task is to:
#     1. Perform Test Classification using Multinomial Naive Bayes(already implemented in sklearn).
#     2. Implement Naive Bayes on your own from scratch for text classification. 
#     3. Compare Results of your implementation of Naive Bayes with one in Sklearn.
# #### Dataset - 
#     http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
# #### Comments : 
#     Your code must have proper comments for better understanding.
# #### Score : 
#     Score will be given by the TA based on your submission.
# #### Submission : 
#     You have to upload zipped file which has python notebook with implementation and dataset used.
# #### Your project will be evaluated on following parameters -
#     1. Correctness of Code - Own Implementation Naive Bayes (Max Score 50)
#     2. Comparison (Max Score 10)
#     3. Commenting (Max Score 10)
#     4. Correctness of Code - Sklearn Naive Bayes (Max Score 30
# 

# In[10]:



# coding: utf-8

# Data Cleaning


import re
headers_list = []

s = ""

with open("headers_info.txt") as headers_file :
    s += ''.join(headers_file.readlines())

words = s.split(' ')

for word in words :
    if re.search("\w\:", word) != None:
        headers_list.append(word)
headers_list[headers_list.index('Writeto:')] = 'Write to:'
print(headers_list)




def preprocess_corpus(corpus, headers_list):
    
    from nltk.corpus import stopwords
    from spacy.lang.en.stop_words import STOP_WORDS
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

    ENGLISH_STOP_WORDS_LIST = list(ENGLISH_STOP_WORDS)
    STOP_WORDS_LIST = list(STOP_WORDS)

    stop_words = list(set(stopwords.words('english') + ENGLISH_STOP_WORDS_LIST + STOP_WORDS_LIST))
    
    #print(stop_words)
    headers_removed_corpus = []
    for line in corpus:
        #line = line.lower()
        line = line.strip()
        if line == '' :
            continue
        is_header = False
        for header in headers_list:
            if header in line:
                is_header = True
        if not is_header :
            headers_removed_corpus.append(line)
        #else:
        #    print(line)
    #pprint(headers_removed_corpus)
    
    import copy
    headers_removed_corpus = headers_removed_corpus #copy.deepcopy(corpus)
    import string
    def to_english(s):
        arr = re.sub('\s', ' ', s)
        arr = arr.split(' ')
        retval = []
        for word in arr:
            if word.translate(string.punctuation).isalnum() : 
                retval.append(word.strip())
        return ' '.join(retval)

    non_english_and_punct_removed = []
    for line in headers_removed_corpus:
        clean_line = []
        #print(line)
        line = re.sub("[:,-]", ' ', line) #,'!"#$%&()*,-.:;<=>?@^_`{|}~'
        line = re.sub("[!\"#$%&\'()\*\+,\-\./:;<=>?@\[\\\]^_`{|}~]", ' ', line) #,'!"#$%&()*,-.:;<=>?@^_`{|}~'
        #print(line)
        words = line.split(' ')
        for word in words:
            clean_line.append(to_english(word).strip())
        clean_line = (' '.join(clean_line)).strip()
        clean_line = re.sub('\s +', ' ', clean_line)

        if clean_line != '':
            non_english_and_punct_removed.append(clean_line)
    #pprint(non_english_and_punct_removed)

    # remove stopwords

    stop_words_removed = []
    for line in non_english_and_punct_removed:
        words = line.split(' ')
        new_line = []
        for word in words:
            word = re.sub("[0-9]+", '', word)
            word = re.sub("\s", ' ', word)
            word = word.strip().lower()
            if word == '' :
                continue
            if word not in stop_words:
                new_line.append(word)
        new_line = ' '.join(new_line)
        if new_line != '' :
            stop_words_removed.append(new_line)
    #pprint(stop_words_removed)

    final_data = '.'.join(stop_words_removed)
    return final_data


# In[5]:


document_paths = []
from pprint import pprint
import os
walk = os.walk('.\\20_newsgroups', topdown = False)
for root, dirs, files in walk :
    for file in files:
        doc = {}
        doc['root'] = root
        doc['file'] = file
        document_paths.append(doc)
#pprint(document_paths[0:100])

corpus = ""
i = 0
import time
st = time.time()
new_paths = []
for doc_path in document_paths:
    #if i == 2 :
    #    break
    path = doc_path['root'] + "\\" + doc_path['file']
    with open(path) as doc :
        data = doc.readlines()
        i += 1
        clean_corpus = preprocess_corpus(data, headers_list)
        clean_data_file_root = doc_path['root'].replace('.\\','.\\clean_data\\')
        os.makedirs(clean_data_file_root, exist_ok = True)
        clean_data_file_path = clean_data_file_root + "\\" +doc_path['file'] + '.txt'
        with open(clean_data_file_path, 'wb') as file_clean_data :
            file_clean_data.write(bytes(clean_corpus,'utf8'))
            file_clean_data.close()
        doc.close()
        if i % 1000 == 0 :
            print( i, "Files Processed in", round(time.time() - st, 3), "sec")
            st = time.time()

#pprint(new_paths)


# In[11]:



# coding: utf-8

# In[16]:


from pprint import pprint
from nltk.corpus import words as nltk_words

english_dictionary = {word : True for word in list(set(nltk_words.words()))}

def is_in_english(word):
    try:
        return english_dictionary[word]
    except:
        return False


# In[17]:


"""
Build Dictionary of Words
"""
document_paths = []
from pprint import pprint
import os
walk = os.walk('.\\clean_data\\20_newsgroups', topdown = False)
for root, dirs, files in walk :
    for file in files:
        doc = {}
        dir_ = root.split('\\')[-1]
        doc['path'] = root+"\\"+file
        doc['target'] = dir_
        document_paths.append(doc)
#pprint(document_paths[0:10000:100])


# In[18]:


import pandas as pd
df = pd.DataFrame(document_paths)
#df.head()


# In[6]:


targets = list(set(df['target'].values))
pprint(targets)
class_dict = { i : targets[i] for i in range(len(targets))}
pprint(class_dict)

class_df = pd.DataFrame()
class_df['class'] = class_dict
class_df['category'] = [class_dict[key] for key in class_dict]
class_df.to_csv("class_dict.csv", index = False)

df['class'] = [targets.index(target) for target in df['target'].values]

df.head()

X = df['path']
Y = df['class']
print(X[0:5])
print(Y[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

train_df = pd.DataFrame()
train_df['path'] = X_train
train_df['class'] = Y_train

train_df.to_csv("train_files.csv", index = False)

test_df = pd.DataFrame()
test_df['path'] = X_test
test_df['class'] = Y_test
test_df.to_csv("test_files.csv", index = False)

train_df.head()

train_df.head()

test_df.head()

master_dictionary = {}

i = 0
import time
st = time.time()
new_paths = []
for doc_path in X_train:
    #if i == 1 :
    #    break
    path = doc_path
    with open(path) as doc :
        num_tokens_in_doc = 0
        data = ''.join(doc.readlines()).split('.')
        for line in data:
            for word in line.split():
                if not is_in_english(word):
                    continue
                #print(word)
                try:
                    master_dictionary[word] += 1
                except KeyError:
                    master_dictionary[word] = 1
                    continue
        doc.close()
        i += 1
        if i%500 == 0 :
            print( i, "Files Processed in", int((time.time() - st)*1000) , "millisec")
            st = time.time()
#print(master_dictionary)

import numpy as np
keys = list(master_dictionary)
freq = np.array([ master_dictionary[key] for key in keys])

words_df = pd.DataFrame()
words_df['word'] = keys
words_df['frequency'] = freq
words_df.to_csv('all_words.csv', index = False)


# In[12]:



# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
words_df = pd.DataFrame(pd.read_csv("all_words.csv"))

words_df.head()

all_words = words_df['word']
freq = words_df['frequency']
master_dictionary = {all_words[i] : freq[i] for i in range(len(freq))}

print(max(freq))

max_allowed_freq = max(freq)
min_allowed_freq = 15

filtered_freq = [f for f in freq if (f <= max_allowed_freq and f >= min_allowed_freq)]
print(len(filtered_freq))
import matplotlib.pyplot as plt
plt.hist(filtered_freq, 50)
plt.show()

too_common_words = [word for word in master_dictionary if master_dictionary[word] > max_allowed_freq]
print("No. of common words : ", len(too_common_words))


too_rare_words = [word for word in master_dictionary if master_dictionary[word] < min_allowed_freq]
print("No. of rare words : ", len(too_rare_words))
#print(too_rare_words)

import matplotlib.pyplot as plt
Y_freq = freq[freq >= min_allowed_freq]
Y_freq = Y_freq[Y_freq <= max_allowed_freq]
plt.plot(np.sort(Y_freq))
plt.show()

selected_words = [word for word in all_words if master_dictionary[word] >= min_allowed_freq 
                       and master_dictionary[word] <= max_allowed_freq 
                       and len(word) > 1]
#for word in keys:
#    if freq
print(len(selected_words))
print(selected_words)

selected_words_freq = [master_dictionary[word] for word in selected_words]
plt.plot(np.sort(selected_words_freq))
plt.show()

selected_words_df = pd.DataFrame()
selected_words_df['word'] = selected_words
selected_words_df['frequency'] = selected_words_freq
selected_words_df.to_csv('selected_words.csv', index = False)


# In[13]:



# coding: utf-8

"""
Build Dataset
"""
def prepare_dataset(selected_words_file_path, data_files_path):
    import numpy as np
    import pandas as pd

    class_dict_df = pd.DataFrame(pd.read_csv("class_dict.csv"))
    #class_dict_df.head()

    #selected_words_df = pd.DataFrame(pd.read_csv('all_words.csv'))
    selected_words_df = pd.DataFrame(pd.read_csv(selected_words_file_path))
    #selected_words_df.head()

    selected_words = selected_words_df['word'].values
    num_selected_words = len(selected_words)

    data_file_paths_df = pd.DataFrame(pd.read_csv(data_files_path))
    #train_file_paths_df.head()

    data_file_paths = data_file_paths_df['path'].values
    data_file_class = data_file_paths_df['class'].values

    import time
    matrix = []
    st = time.time()
    for i in range(len(data_file_paths)) :
        #if i == 2000:
        #    break
        if i % 1000 == 0 :
            et = time.time()
            print( i, "Files Processed in", int((time.time() - st)*1000) , "millisec")
            st = time.time()    
        path = data_file_paths[i]
        target = data_file_class[i]
        with open(path) as file :
            X = []
            count = {}
            data = file.readlines()
            words_in_file = (''.join(data)).split(' ')
            for word in words_in_file:
                if word in selected_words:
                    try:
                        count[word] += 1
                    except:
                        count[word] = 1
            for word in selected_words:
                try :
                    x = count[word] #X.append(count[word])
                except :
                    count[word] = 0 #X.append(0)
            #print(X)
            X = [count[word] for word in selected_words if word in count] 
            #print(len(X))
            matrix.append(X)
            #X_train_df.iloc[i, :] = 
            file.close()
    print(len(matrix))
    print(len(matrix[0]))

    X_df = pd.DataFrame(matrix, columns = selected_words)

    #X_df.describe()

    import copy
    X = X_df.values
    Y = copy.deepcopy(data_file_class)
    print(X.shape)
    print(Y.shape)

    dataset_df = pd.DataFrame(X_df)
    dataset_df['target'] = Y
    return dataset_df

train_files_path = "train_files.csv"
test_files_path = "test_files.csv"

selected_words_file_path = 'selected_words.csv'

train_dataset_df = prepare_dataset(selected_words_file_path, train_files_path)
train_dataset_df.to_csv("20_newsgroups_dataset.csv", index = False)

test_dataset_df = prepare_dataset(selected_words_file_path, test_files_path)
test_dataset_df.to_csv("20_newsgroups_test_dataset.csv", index = False)


# In[14]:



# coding: utf-8

import numpy as np
import pandas as pd

import time
st = time.time()
train_dataset_df = pd.read_csv("20_newsgroups_dataset.csv")
et = time.time()

print("Loading Time for Training Set:", round((et-st)*1000), "millisecond")

train_dataset_df.describe()

X_train_df = train_dataset_df.drop('target', axis = 1) 
Y_train_df = train_dataset_df['target']

# X_train_df.describe()

# Y_train_df.describe()

import time
st = time.time()
test_dataset_df = pd.read_csv("20_newsgroups_test_dataset.csv")
et = time.time()

print("Loading Time for Test Set:", round((et-st)*1000), "millisecond")

test_dataset_df.describe()

X_test_df = test_dataset_df.drop('target', axis = 1) 
Y_test_df = test_dataset_df['target']

# X_test_df.describe()

# Y_test_df.describe()

words = X_train_df.columns
#print(words)
possible_classes = list(set(Y_train_df.values))
#print(possible_classes)

def predict_single(X, model) :
    
    max_class_prob = -np.inf
    max_class = None
    possible_classes = model.keys()
    for y in possible_classes :
        X = np.array(X)
        prob_X_equals_x_given_Y_equals_y = model[y]['class_prior']
        prob_X_equals_x_given_Y_equals_y += (X * model[y]['log_prob_sum']).sum()
        if prob_X_equals_x_given_Y_equals_y > max_class_prob :
            max_class_prob = prob_X_equals_x_given_Y_equals_y
            max_class = y
    return max_class
    
def generate_model(dictionary, alpha) :
    model = {}
    possible_classes = dictionary['possible_classes']
    num_words = dictionary["vocabulary_size"]
    
    range_words = range(num_words)
    
    for y in possible_classes :
        #class_prior
        model[y] = {}
        prob_Y_equals_y = np.log(dictionary[y]["class_count"]/dictionary["total_data"])
        prob_X_equals_x_given_Y_equals_y = 0 
        prob_X_equals_x_given_Y_equals_y += prob_Y_equals_y
        total_words_in_class_y_docs = dictionary[y]["total_words"]        
        count_f_class_y = np.array([dictionary[y][f] for f in range_words])
        prob_f_class_y = (count_f_class_y+alpha)/(total_words_in_class_y_docs + alpha*num_words) 
        
        model[y]['class_prior'] = prob_Y_equals_y
        model[y]['log_prob_sum'] = np.log(prob_f_class_y)
    
    return model
    

def predict(X, model) :
    #if not isinstance(X[0], list) :
    #    return None #predict_single(X, model)
    Y_pred = []
    for x in X :
        Y_pred.append(predict_single(x, model))
    return Y_pred

def fit(X, Y, alpha) :
    if alpha > 1 or alpha < 0 :
        print("Alpha parameter not in range [0,1]...")
        print("setting alpha = 1")
        alpha = 1
    num_words = len(X[0])
    possible_classes = list(set(Y))
    dictionary = {}
    dictionary["total_data"] = len(Y)
    dictionary["vocabulary_size"] = num_words
    dictionary["possible_classes"] = possible_classes
    for y in possible_classes :
        y_dict = {i : X[Y == y, i].sum() for i in range(num_words)}
        y_dict['total_words'] = sum(y_dict.values())
        y_dict['class_count'] = sum(Y == y)
        #print(y_dict['total_count'])
        dictionary[y] = y_dict

    return generate_model(dictionary, alpha)

import time
st = time.time()
model = fit(X_train_df.values, Y_train_df.values, 0.01)
et = time.time()

print("Training Time:", round((et-st)*1000), "millisecond")

#print(model)

"""
Train Data Evaluation
"""
import time
st = time.time()
Y_train_pred = predict(X_train_df.values, model)
et = time.time()

print("Prediction Time for Training Set:", round((et-st)*1000), "millisecond")

#print(Y_train_pred)

from sklearn.metrics import confusion_matrix

from pprint import pprint
pprint(confusion_matrix(Y_train_pred, Y_train_df.values))

from sklearn.metrics import classification_report
print(classification_report(Y_train_pred, Y_train_df.values))

"""
Test Data Evaluation
"""
import time
st = time.time()
Y_test_pred = predict(X_test_df.values, model)
et = time.time()

print("Prediction Time for Testing Set:", round((et-st)*1000), "millisecond")

training_time = (et-st)
print(training_time)

from sklearn.metrics import confusion_matrix

from pprint import pprint
pprint(confusion_matrix(Y_test_pred, Y_test_df.values))

from sklearn.metrics import classification_report
print(classification_report(Y_test_pred, Y_test_df.values))


# In[19]:


from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import time
st = time.time()
sklearn_model = MultinomialNB(alpha = 0.01)
sklearn_model.fit(X_train_df.values, Y_train_df.values)
et = time.time()
print("Training Time:", round((et-st)*1000), "millisecond")


import time
st = time.time()
Y_train_pred = sklearn_model.predict(X_train_df.values)
et = time.time()
print("Prediction Time for Training Set:", round((et-st)*1000), "millisecond")

pprint(confusion_matrix(Y_train_pred, Y_train_df.values))
print(classification_report(Y_train_pred, Y_train_df.values))

import time
st = time.time()
Y_test_pred = sklearn_model.predict(X_test_df.values)
et = time.time()
print("Prediction Time for Test Set:", round((et-st)*1000), "millisecond")

pprint(confusion_matrix(Y_test_pred, Y_test_df.values))
print(classification_report(Y_test_pred, Y_test_df.values))

