<<<<<<< HEAD
import sklearn
import numpy as np


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

train_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino")

data_train = train_set.data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Training a classfier

clf = SGDClassifier().fit(X_train_tfidf, train_set.target)

#Testing with 'docs_new'

    #docs_new = ['Violao Tagima preço 2000', 'violao bateria eletrônica preço 2000']
    #X_new_counts = count_vect.transform(docs_new)
    #X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    #predicted = clf.predict(X_new_tfidf)

    #for doc, category in zip(docs_new, predicted):
    #    print('%r => %s' % (doc, train_set.target_names[category]))

#Evaluation of the test

test_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino")

docs_test = test_set.data

X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

print('Accuracy is %.2f' % (np.mean(predicted == test_set.target)*100))




=======
import sklearn
import numpy as np


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

train_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino")

data_train = train_set.data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Training a classfier

clf = SGDClassifier().fit(X_train_tfidf, train_set.target)

docs_new = ['Violao Tagima preço 2000', 'Bateria eletrônica preço 2000']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(docs_new, predicted):
#    print('%r => %s' % (doc, train_set.target_names[category]))

#Evaluation of the test

test_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino")

docs_test = test_set.data

X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

print('Accuracy is %.2f' % (np.mean(predicted == test_set.target)*100))




>>>>>>> a46318c1622912e12da40ff79b6f630859f39e94
