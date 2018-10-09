import sklearn
import numpy as np
import pandas
import time

from nltk import ngrams, FreqDist
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import soups_of_interest
from utils import clean_text
from tqdm import tqdm
import csv
import os 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
        labels = ["Documentos"] + words + ["Rotulo"]
        writer.writerow(labels)
        row_count, column_count = 0, 0
        path_to_texto_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\texto\""
        path_to_texto = path_to_texto_aux[:-1]
        for arq in tqdm(os.listdir(path_to_texto)):
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
    words_stem = []
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
    storeBoW("BoW_plain", pag_tokens, list(words))

    #Convertendo os tokens das palavras para lowercase
    pag_tokens_lower = pag_tokens.copy()
    words_lower = list(words)
    for c in pag_tokens_lower:
        for x in range(len(pag_tokens_lower[c])):
            pag_tokens_lower[c][x] = (pag_tokens_lower[c][x]).lower()
    for p in range(len(words_lower)):
        words_lower[p] = (words_lower[p]).lower()
    storeBoW("BoW_lowercase", pag_tokens_lower, words_lower)

    #Criando Bag of Words sem stopwords
    stopWords_p = stopwords.words('portuguese')
    stopWords_e = stopwords.words('english')
    stopWords_p = set(stopWords_p)
    stopWords_e = set(stopWords_e)
    pag_tokens_stopwords = pag_tokens_lower.copy()
    for c in pag_tokens_stopwords:
        pag_tokens_stopwords[c] = list(set(pag_tokens_stopwords[c]) - stopWords_p - stopWords_e)
    words_stopwords = set(words_lower) - stopWords_p - stopWords_e
    words_stopwords = list(words_stopwords)
    print("Stopwords:", len(words_stopwords))
    storeBoW("BoW_stopwords", pag_tokens_stopwords, words_stopwords)

    #Criando Bag of Words com stemming
    pag_tokens_stemming = pag_tokens_stopwords.copy()
    for c in pag_tokens_stopwords:
        for p in range(len(pag_tokens_stopwords[c])):
            pag_tokens_stopwords[c][p] = PorterStemmer().stem(pag_tokens_stopwords[c][p])
        pag_tokens_stopwords[c] = list(set(pag_tokens_stopwords[c]))
    for p in words_stopwords:
        words_stem.append(PorterStemmer().stem(p))
    words_stem = list(set(words_stem))
    storeBoW("Bow_stemming", pag_tokens_stopwords, words_stem)

#Treinamento e Avaliação dos modelos de classificação
def createResultcsvFiles(modelo):
    path_to_results_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\resultados\""
    path_to_results = path_to_results_aux[:-1]

    with open(path_to_results+modelo+"_"+".csv", "w+", encoding="utf8", newline='') as csvFile:
        writer = csv.writer(csvFile)
        labels = ["", "Accuracy", "Precision", "Recall", "Training Time"]
        writer.writerow(labels)

def saveMeasurements(accuracy, precision, recall, trainTime, bow, modelo):
            path_to_results_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\resultados\""
            path_to_results = path_to_results_aux[:-1]

            with open(path_to_results+modelo+"_"+".csv", "a+", encoding="utf8", newline='') as csvFile:
                writer = csv.writer(csvFile)
                row = [bow, accuracy.mean(), precision.mean(), recall.mean(), trainTime]
                writer.writerow(row)

def evaluation(clf, x, y, modelo, bow):
        path_to_bow_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\bag_of_words\""
        path_to_bow = path_to_bow_aux[:-1]
        print(modelo)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
        accuracy = cross_val_score(clf, x, y, cv=15)
        print("Accuracy: %0.2f" % accuracy.mean())

        precision = cross_val_score(clf, x, y, cv=15, scoring='precision')        
        print("Precision: %0.2f" % precision.mean())
        
        recall = cross_val_score(clf, x, y, cv=15, scoring='recall')
        print("Recall: %0.2f" % recall.mean())

        start_time = time.time()
        clf.fit(x_train, y_train)
        end_time = time.time()
        trainTime = end_time - start_time
        print("Train Time: %g segundos" % trainTime)

        bow_plaintext = path_to_bow+"BoW_plain.csv"
        bow_lowercase = path_to_bow+"BoW_lowercase.csv"
        bow_stopwords = path_to_bow+"BoW_stopwords.csv"

        if(bow == bow_plaintext):
            saveMeasurements(accuracy, precision, recall, trainTime, "plain", modelo)
        elif(bow == bow_lowercase):
            saveMeasurements(accuracy, precision, recall, trainTime, "lowercase", modelo)
        elif(bow == bow_stopwords):
            saveMeasurements(accuracy, precision, recall, trainTime, "stopword", modelo)
        else:
            saveMeasurements(accuracy, precision, recall, trainTime, "stemming", modelo)

def saveVectorizer(predicted, test_set, modelo, feature, shape, trainTime):
    path_to_results_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\resultados\""
    path_to_results = path_to_results_aux[:-1]

    arq = open(path_to_results+modelo+"_vectorizer_"+feature+".txt", "w+", encoding="utf8")
    arq.write("Modelo: %s \n" % modelo)
    arq.write("Feature: %s \n" % feature)
    arq.write("Shape: %s \n" % str(shape))
    arq.write('Accuracy is %.2f \n' % (np.mean(predicted == test_set.target)*100))
    arq.write('Train Time is %f \n' % trainTime)
    arq.write(metrics.classification_report(test_set.target, predicted, target_names=test_set.target_names))
    arq.close

def trainVectorizer():
    train_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Treino", random_state=42)
    data_train = train_set.data

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    test_set = sklearn.datasets.load_files(container_path=r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\Teste\html", random_state=42)
    data_test = test_set.data

    X_test_counts = count_vect.transform(data_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    #Train vectorizer
    start_time = time.time()
    clf_MLP_tf = MLPClassifier(hidden_layer_sizes=(10, 5), solver='lbfgs').fit(X_train_tfidf, train_set.target)
    end_time= time.time()
    train_time_mlp_tf = end_time - start_time

    start_time = time.time()
    clf_multinomial_tf = MultinomialNB().fit(X_train_tfidf, train_set.target)
    end_time = time.time()
    train_time_multinomial_tf = end_time - start_time

    start_time = time.time()
    clf_gaussian_tf = GaussianNB().fit(X_train_tfidf.toarray(), train_set.target)
    end_time = time.time()
    train_time_gaussian_tf = end_time - start_time

    start_time = time.time()
    clf_rf_tf = RandomForestClassifier(n_estimators = 100).fit(X_train_tfidf, train_set.target)
    end_time = time.time()
    train_time_rf_tf = end_time - start_time

    start_time = time.time()
    clf_lr_tf = linear_model.LogisticRegression().fit(X_train_tfidf, train_set.target)
    end_time = time.time()
    train_time_lr_tf = end_time - start_time

    #Predict
    predicted_MLP_tf = clf_MLP_tf.predict(X_test_tfidf)
    predicted_multinomial_tf = clf_multinomial_tf.predict(X_test_tfidf)
    predicted_gaussian_tf = clf_gaussian_tf.predict(X_test_tfidf.toarray())
    predicted_rf_tf = clf_rf_tf.predict(X_test_tfidf)
    predicted_lr_tf = clf_lr_tf.predict(X_test_tfidf)

    #Salvando resultados
    saveVectorizer(predicted_MLP_tf, test_set, "MLP", "TF-IDF", X_train_tfidf.shape, train_time_mlp_tf)
    saveVectorizer(predicted_multinomial_tf, test_set, "MultinomialNB", "TF-IDF", X_train_tfidf.shape, train_time_multinomial_tf)
    saveVectorizer(predicted_gaussian_tf, test_set, "GaussianNB", "TF-IDF", X_train_tfidf.shape, train_time_gaussian_tf)
    saveVectorizer(predicted_rf_tf, test_set, "RandomForest", "TF-IDF", X_train_tfidf.shape, train_time_rf_tf)
    saveVectorizer(predicted_lr_tf, test_set, "LogisticRegression", "TF-IDF", X_train_tfidf.shape, train_time_lr_tf)

def trainModels(): 
    path_to_bow_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\bag_of_words\""
    path_to_bow = path_to_bow_aux[:-1]
    path_to_results_aux = r"C:\Users\Lucas\Documents\EC\10º Período\RI\RI-part1\data\resultados\""
    path_to_results = path_to_results_aux[:-1]

    bow_plaintext = path_to_bow+"BoW_plain.csv"
    bow_lowercase = path_to_bow+"BoW_lowercase.csv"
    bow_stopwords = path_to_bow+"BoW_stopwords.csv"
    bow_stemming = path_to_bow+"BoW_stemming.csv"
    BoWs = [bow_plaintext, bow_lowercase, bow_stopwords, bow_stemming]

    createResultcsvFiles("MultinomialNB")
    createResultcsvFiles("GaussianNB")
    createResultcsvFiles("MLP")
    createResultcsvFiles("RandomForest")
    createResultcsvFiles("LogisticRegression")

    for arq in BoWs:
        words = pandas.read_csv(arq, engine='python')
        rotulos = pandas.read_csv(arq, engine='python')
        linha0 = []
        for a in tqdm(words):
            linha0.append(a)
        del words['Documentos']
        del words['Rotulo']
        for i in tqdm(linha0):
            if(i != 'Rotulo'):
                del rotulos[i]

        words = words.values
        rotulos = rotulos.values
        rotulos = np.ravel(rotulos)
        
        clf_multinomial = MultinomialNB()        
        clf_gaussian = GaussianNB()        
        clf_mlp = MLPClassifier(solver='lbfgs', random_state=2, hidden_layer_sizes=(10, 5))        
        clf_rf = RandomForestClassifier(n_estimators = 100)
        clf_lr = linear_model.LogisticRegression()
        
        evaluation(clf_multinomial, words, rotulos, 'MultinomialNB', arq)
        evaluation(clf_gaussian, words, rotulos, 'GaussianNB', arq)
        evaluation(clf_mlp, words, rotulos, 'MLP', arq)
        evaluation(clf_rf, words, rotulos, 'RandomForest', arq)
        evaluation(clf_lr, words, rotulos, 'LogisticRegression', arq)

def main():
    htmlTotext = input("Converter arquivos html para texto?")
    if(htmlTotext == "S"):
        print("Convertendo html para txt ...")
        htmlTotxt()
    
    preproc = input("Realizar pre-processamento e geração das Bag of Words?")
    if(preproc == "S"):
        print("Realizando pre-processamento e gerando Bag of Words ...")
        createBoWs()

    train = input("Treinar modelos e realizar avaliação?")
    if(train == "S"):
        print("Treinando modelos e realizando avaliação ...")
        trainModels()
    train_vect = input("Treinar modelos e realizar avaliação - vectorizer?")
    if(train_vect == "S"):
        print("Treinando - vectorizer ...")
        trainVectorizer()

if __name__ == "__main__":
    main()
