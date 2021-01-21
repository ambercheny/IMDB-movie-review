import pandas as pd
import nltk
import re

### Preprocessing
# Training dataset
train = pd.read_csv('D:/amberstuff/1091/ML/個人競賽/all.csv')
train[['Id','rating']] = train.filename.str.split("_",expand=True,)

# convert all content to lowercase
train["stories_lower"] = train["stories"].str.lower()
   
# tokenization
def identify_tokens(df_name):
    stories_lower = df_name['stories_lower']
    tokens = nltk.word_tokenize(stories_lower)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
train['stories_lower_tok'] = train.apply(identify_tokens, axis=1)

# remove stopwords
def remove_stops(df_name):
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english")) 
    my_list = df_name['stories_lower_tok']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)
train['meaningful_words'] = train.apply(remove_stops, axis=1)
    
# for each value in meaningful_words, it is a list of tokens. In order to do machine learning, we need to convert lists to strings
def rejoin_words(row):
    my_list = row['meaningful_words']
    joined_words = ( " ".join(my_list))
    return joined_words
train['processed'] = train.apply(rejoin_words, axis=1)


# Testing dataset
test_data = pd.read_csv('D:/amberstuff/1091/ML/個人競賽/test_dataset.csv')
test_data=test_data.rename(columns = {'Content': 'stories0'}, inplace = False) # change column name
test_data["stories"] = test_data["Title"]+ " " +test_data["stories0"] # merge column "title" and "stories0" to stories

test_data["stories_lower"] = test_data["stories"].str.lower()

def identify_tokens(row):
    stories_lower = row['stories_lower']
    tokens = nltk.word_tokenize(stories_lower)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
test_data['stories_lower_tok'] = test_data.apply(identify_tokens, axis=1)

def remove_stops(row):
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english")) 
    my_list = row['stories_lower_tok']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)
test_data['meaningful_words'] = test_data.apply(remove_stops, axis=1)

def rejoin_words(row):
    my_list = row['meaningful_words']
    joined_words = ( " ".join(my_list))
    return joined_words
test_data['processed'] = test_data.apply(rejoin_words, axis=1)



### Build models
# MultinomialNB - stories/Content
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

model = MultinomialNB()
kf=KFold(n_splits=20, shuffle=True)
tfidf_vectorizer = TfidfVectorizer()

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.stories):
    x_train = np.array(train_data.stories)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.stories)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data["stories"]),
                             train_data['Label'])
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['Content'].values.astype('U')))

# MultinomialNB - processed/Content
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

model = MultinomialNB()
kf=KFold(n_splits=20, shuffle=True)
tfidf_vectorizer = TfidfVectorizer()

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.processed):
    x_train = np.array(train_data.processed)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.processed)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data["processed"]),
                             train_data['Label'])
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['Content'].values.astype('U')))


# knn - stories/processed
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn import metrics
from sklearn import datasets
model = KNeighborsClassifier()
kf=KFold(n_splits=3, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.stories):
    x_train = np.array(train_data.stories)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.stories)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data.stories),
                             train_data.Label)
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['processed']))


# decision tree - stories/processed
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn import metrics
from sklearn import datasets
model = DecisionTreeClassifier()
kf=KFold(n_splits=20, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.stories):
    x_train = np.array(train_data.stories)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.stories)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data.stories),
                             train_data.Label)
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['processed']))


# Logistic Regression - stories/processed
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn import metrics
from sklearn import datasets
model = LogisticRegression()
kf=KFold(n_splits=3, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.stories):
    x_train = np.array(train_data.stories)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.stories)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data.stories),
                             train_data.Label)
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['processed']))


# SVM - stories/processed
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn import metrics
from sklearn import datasets
model = SVC()
kf=KFold(n_splits=20, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.stories):
    x_train = np.array(train_data.stories)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.stories)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data.stories),
                             train_data.Label)
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )


# SVM - processed/processed
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn import metrics
from sklearn import datasets
model = SVC()
kf=KFold(n_splits=20, shuffle=True)

predicted= []
expected = []
for train_index, test_index in kf.split(train_data.processed):
    x_train = np.array(train_data.processed)[train_index]
    y_train = np.array(train_data.Label)[train_index]
    x_test = np.array(train_data.processed)[test_index]
    y_test = np.array(train_data.Label)[test_index]
    
    model.fit(tfidf_vectorizer.fit_transform(train_data.processed),
                             train_data.Label)
    expected.extend(y_test) 
    predicted.extend(model.predict(tfidf_vectorizer.transform(x_test)))
    
print(metrics.classification_report(expected, predicted))

print("Macro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='macro'),
    metrics.recall_score(expected, predicted, average='macro'),
    metrics.f1_score(expected, predicted, average='macro'))
    )
print("Micro-Avg PRF: {0}, {1}, {2}".format(
    metrics.precision_score(expected, predicted, average='micro'),
    metrics.recall_score(expected, predicted, average='micro'),
    metrics.f1_score(expected, predicted, average='micro'))
    )

test_data["Label"]=model.predict(tfidf_vectorizer.transform(test_data['processed']))
