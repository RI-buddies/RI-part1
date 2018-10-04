import sklearn
import numpy as np

from nltk import ngrams, FreqDist
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import soups_of_interest
from utils import clean_text
import csv
import os 


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

#Extração de texto das páginas html
def htmlTotxt():
    i = 0
    for arq in os.listdir(r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\html"):
        if(arq.endswith(".html")):
            i += 1
            path_html_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\html\""
            path_html = path_html_aux[:-1]
            arquivo = open(path_html+arq, "r+", encoding="utf8")
            pag = arquivo.read()
            arquivo.close()              
            path_out_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\texto\""
            path_out = path_out_aux[:-1]
            out = open(path_out+arq[:-4]+"txt", "w+", encoding="utf8")
            out.write(soups_of_interest(pag))
            out.close
    return


## Pré Processamento

#Função que armazena a Bag of Words num arquivo .cvs
def storeBoW(name, texts, words): 
    path_to_bow_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\bag_of_words\""
    path_to_bow = path_to_bow_aux[:-1]
    with open(path_to_bow+name+'.csv', 'w', encoding="utf8",  newline='') as csvFile:
        writer = csv.writer(csvFile)
        labels = ["Documentos"] + words + ['Rótulo']
        writer.writerow(labels)
        row_count, column_count = 0, 0
        path_to_texto_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\texto\""
        path_to_texto = path_to_texto_aux[:-1]
        for arq in os.listdir(path_to_texto):
            if(arq.endswith(".txt")):
                row_count += 1
                row = [str(arq[:-6])]
                for word in words:
                    if(word in texts[arq]):
                        row.append(1)
                    else:
                        row.append(0)
                if(arq.endswith("p.txt")):  
                    row.append(1)
                elif(arq.endswith("n.txt")):
                    row.append(0)
                writer.writerow(row)
                column_count = len(row)
                print("Tamanho da bag of words: (%d, %d)" %(row_count, column_count))            


# Criando Bag of Words de todo o Texto
def createBoWs():
    words = set()
    pag_tokens = {}
    path_to_texto_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\texto\""
    path_to_texto = path_to_texto_aux[:-1]

    #Criando Bag of Words plainText
    for arq in os.listdir(r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\texto"):
        if(arq.endswith(".txt")):
            arquivo = open(path_to_texto+arq, "r+", encoding="utf8")
            pag = arquivo.read()
            arquivo.close()
            pag = clean_text(pag)
            tokenizadas = set(word_tokenize(pag))
            words = words.union(tokenizadas)
            pag_tokens[arq] = word_tokenize(pag)
    # arquivo = open(r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\plain-text.txt", "w", encoding="utf8")
    # arquivo.write(str(list(words)))
    # arquivo.close()
    storeBoW("BoW_plain", pag_tokens, list(words))

    #Criando Bag of Words lowercase
    pag_tokens_lower = pag_tokens
    words_lower = list(words)
    for c in pag_tokens_lower:
        for x in range(len(pag_tokens_lower[c])):
            pag_tokens_lower[c][x] = (pag_tokens_lower[c][x]).lower()
    for p in range(len(words_lower)):
        words_lower[p] = (words_lower[p]).lower()
    storeBoW("BoW_lowercase", pag_tokens_lower, words_lower)

    #Criando Bag of Words stopwords
    stopWords = stopwords.words('portuguese')
    stopWords = set(stopWords)
    pag_tokens_stopwords = pag_tokens_lower
    for c in pag_tokens_stopwords:
        pag_tokens_stopwords[c] = list(set(pag_tokens_stopwords[c]) - stopWords)
    words_stopwords = set(words_lower) - stopWords
    words_stopwords = list(words_stopwords)
    storeBoW("BoW_stopwords", pag_tokens_stopwords, words_stopwords)

#Treinando modelos de classificação
def trainModels(): #FINALIZAR
    path_to_bow_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\bag_of_words\""
    path_to_bow = path_to_bow_aux[:-1]
    bow_plaintext = path_to_bow+"BoW_plain.csv"
    bow_lowercase = path_to_bow+"BoW_lowercase.csv"
    bow_stopwords = path_to_bow+"BoW_stopwords.csv"
    BoWs = [bow_plaintext, bow_lowercase, bow_stopwords]
       

    train_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino")


    data_train = train_set.data

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #Training a classfier

    #clf = MLPClassifier().fit(X_train_tfidf, train_set.target)
    clf = MultinomialNB().fit(X_train_tfidf, train_set.target)
    #clf = SGDClassifier().fit(X_train_tfidf, train_set.target)

    #Testing with 'docs_new'

    # docs_new = ['Violao preço 2000', 'gianinni bateria eletrônica preço 2000', 'guitarra elétrica com correia preço 5000']
    # X_new_counts = count_vect.transform(docs_new)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # predicted = clf.predict(X_new_tfidf)

    # for doc, category in zip(docs_new, predicted):
    #     print('%r => %s' % (doc, train_set.target_names[category]))

    #Evaluation of the test

    test_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Teste")

    docs_test = test_set.data

    X_test_counts = count_vect.transform(docs_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)

    # i = 0
    # for doc, category in zip(docs_test, predicted):
    #    print('Doc_%d => %s -- %s' % (i, train_set.target_names[category], test_set.target_names[test_set.target[i]]))
    #    i += 1


    # print('Accuracy is %.2f' % (np.mean(predicted == test_set.target)*100))



