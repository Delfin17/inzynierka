import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
lemmatizerEngine = WordNetLemmatizer()
stemmerEngine = PorterStemmer()
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import numpy as np
import pickle

#ztest = np.zeros(749659)

def LoadDataFromFile(path):
    file = open(path, "r", encoding='utf-8', errors='ignore')
    rawData = file.readlines()
    file.close()
    daneUczace = []

    for line in rawData:     
        line = line.split('","')
        if len(line) == 6:
            tmp=[]
            line[5] = line[5].replace('"\n', '')
            line[0] = line[0].replace('"', '')
            tmp.append(line[5])

            if line[0] == "0":
                tmp.append("0")
            elif line[0] == "4":
                tmp.append("1")

            if tmp[len(tmp)-1] == "0" or tmp[len(tmp)-1] == "1":
                daneUczace.append(tmp)

    return daneUczace

def Tokenizacja(data):
    daneUczace = []
    licznik = 0
    for row in data:
        licznik = licznik + 1

        if licznik % 1000 == 0 and licznik != 0:
            print("Tokenizacja: " + str(licznik) + "\n")

        row[0] = row[0].lower()
        row[0] = nltk.tokenize.word_tokenize(row[0])
        temp = []
        for word in row[0]:
            if len(word) > 2:
                word = lemmatizerEngine.lemmatize(word)
                word = stemmerEngine.stem(word)
                temp.append(word)
        row[0] = temp
        daneUczace.append(row)
    return daneUczace

def CreateWordMap(data):
    wordMap = {}
    counter = 0
    for row in data:
        for word in row[0]:
            if word not in wordMap:
                wordMap[word] = counter
                counter = counter + 1
    return wordMap

def CreateFeatureVector(data, wordMap):
    Vector = []
    for row in data:
        temp = np.zeros(len(wordMap) + 1)
        for word in row[0]:
            index = wordMap[word]
            temp[index] = temp[index] + 1
        temp[len(temp) - 1] = row[1]
        Vector.append(temp)
    return Vector

def CreateFeatureVectorAndPartialFit(data, wordMap, model):
    Vector = []
    licznik = 0
    for row in data:
        temp = np.zeros(len(wordMap) + 1)
        licznik = licznik + 1
        for word in row[0]:
            index = wordMap[word]
            temp[index] = temp[index] + 1
        temp[len(temp) - 1] = row[1]
        Vector.append(temp)
        if licznik % 1000 == 0:
            Vector = np.array(Vector)
            X = Vector[:,:-1]
            Y = Vector[:,-1]
            Vector = []
            print("\n Rozpoczyma uczenie modelu, wiersze do: " + str(licznik) + " wierszy")
            model.partial_fit(X,Y, classes = [0,1])                    

    if licznik % 1000 != 0:
        Vector = np.array(Vector)
        X = Vector[:,:-1]
        Y = Vector[:,-1]
        Vector = []
        print("\n Rozpoczyma uczenie modelu, ostatnie: " + str(licznik) + " wierszy")
        model.partial_fit(X,Y, classes = [0,1])

    return model


def PierwszaWersja(model):
    print("\nRozpoczynam wczytywanie danych: ")
    daneUczace = LoadDataFromFile("Data\dane_uczace.csv")
    print("\nRozpoczynam Tokenizacje")
    daneUczace = Tokenizacja(daneUczace)
    print("\nRozpoczynam tworzenie mapy slow")
    wordMap = CreateWordMap(daneUczace)
    print("\nRozpoczynam tworzenie wekora cech")
    Vector = CreateFeatureVector(daneUczace, wordMap)

    Vector = np.array(Vector)
    #Vector = shuffle(Vector)

    X = Vector[:,:-1]
    Y = Vector[:,-1]

    Xtrain = X[:-100,]
    Ytrain = Y[:-100,]
    Xtest = X[-100:,]
    Ytest = Y[-100:,]

    print("\n\nRozpoczynam trenowanie modelu")
    model.fit(X, Y)
    print("\n\nRozpoczynam dump modelu - pickle")
    pickle.dump(model, open("dumpModelu_pickle.data", 'wb'))
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))


def DrugaWersja(model):
    print("\nRozpoczynam wczytywanie danych: ")
    daneUczace = LoadDataFromFile("Data\dane_uczace.csv")
    print("\nRozpoczynam Tokenizacje")
    daneUczace = Tokenizacja(daneUczace)
    print("\nRozpoczynam tworzenie mapy slow")
    wordMap = CreateWordMap(daneUczace)
    print("\nRozpoczynam tworzenie wekora cech")
    model = CreateFeatureVectorAndPartialFit(daneUczace, wordMap, model)

    print("\n\nRozpoczynam dump modelu - pickle")
    pickle.dump(model, open("dumpModelu_pickle.data_V2", 'wb'))

def TestZapisanegoModelu(nazwaModelu):
    model = pickle.load(open(nazwaModelu, 'rb'))
    daneUczace = LoadDataFromFile("Data\dane_testowe.csv")
    daneUczace = Tokenizacja(daneUczace)
    wordMap = CreateWordMap(daneUczace)
    Vector = CreateFeatureVector(daneUczace, wordMap)

    Vector = np.array(Vector)

    X = Vector[:,:-1]
    Y = Vector[:,-1]

    Xtrain = X[:-100,]
    Ytrain = Y[:-100,]
    Xtest = X[-100:,]
    Ytest = Y[-100:,]

    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))

##PierwszaWersja(MultinomialNB())
DrugaWersja(MultinomialNB())
