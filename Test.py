


import numpy as np
np.random.seed(1337)
from keras.models import load_model
import pandas as pd
import io
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Convolution1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.layers import  merge
from keras.regularizers import l2
from keras.layers import AveragePooling1D
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from Bio import SeqIO
import argparse
import csv

def saveResults(Seq_id, predict):
    fp = open('prediction_results.csv', 'w', newline='')
    a = csv.writer(fp, delimiter=',')
    elements = [['Sequence ID', 'Diffraction-quality Crystal Prediction']]
    a.writerows(elements)



    for i in range(0, len(Seq_id)):
        a = csv.writer(fp, delimiter=',')
        seq = Seq_id[i]
        prediction = str(predict[i][10])
        elements = [[seq, prediction]]
        a.writerows(elements)
    fp.close()
    

    
def ensembleModel(embedTest, mean):
    
    for i in range(0, len(embedTest[:][:])):
        sum = 0
        for j in range(0, 11):
            sum += mean[i][j]
        
        sum = sum/10.0
        mean[i][10] = sum
        
    return(mean)


def predictSeqs(embedTest, weights, jsonFiles):
    mean = np.zeros((len(embedTest[:][:]), 11))
    c = 0
    adam = Adam(lr=0.001)
    for i in range(0, len(weights)):
        json_file = open('DeepCrystal_Models/'+jsonFiles[i], 'r')
        model_json = json_file.read()
        json_file.close()
        load_my_model = model_from_json(model_json)
        load_my_model.load_weights('DeepCrystal_Models/'+weights[i])
        load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        print("Prediction is in progress, please wait...")
        score = load_my_model.predict(embedTest)
        for i in range(0, len(score)):
            mean[i][c] = score[i]
        c = c + 1
    
    print("Prediction is done")
    return (mean)

def cleanSequenes(X_test):
    amino_acids ='ACDEFGHIKLMNPQRSTVWXY'
    for i in range(0, len(X_test)):
        test = []
        st = str(X_test[i])
        testStr =''
        for ch in st:
            if(ch in amino_acids):
                testStr += ch
        
        test.append(testStr)
        X_test[i] = test

    X = []
    for i in range(len(X_test)):
        if(len(X_test[i][0]) <= 800):
            X.append(X_test[i][0])
    X_test = np.array(X)
    X_test = X_test.reshape(len(X_test),1)
    
    embedTest = []
    for i in range(0, len(X_test)):
        length = len(X_test[i][0])
        pos = []
        counter = 0
        st = X_test[i][0]
        for c in st:
            AMINO_INDEX = amino_acids.index(c)
            pos.append(AMINO_INDEX)
            counter += 1
        while(counter < 800):
            pos.append(21)
            counter += 1
        embedTest.append(pos)
            
    return(embedTest)


def parseFastaFile():
    X_test = []
    Seq_id = []
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()

    with args.file as file:
        fasta_sequences = SeqIO.parse(file,'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, fasta.seq.tostring()
            X_test.append(sequence)
            Seq_id.append(name)
            
    return (X_test, Seq_id)

def main():
    
    X_test = []
    Seq_id = []
    X_test, Seq_id = parseFastaFile()
    embedTest = cleanSequenes(X_test)
    
    embedTest = np.array(embedTest)
    mean = np.zeros((len(embedTest[:][:]), 11))
    
    weights = ['model1.hdf5', 'model2.hdf5', 'model3.hdf5', 'model4.hdf5', 'model5.hdf5',
           'model6.hdf5', 'model7.hdf5', 'model8.hdf5', 'model9.hdf5', 'model10.hdf5']
    jsonFiles = ['model1.json', 'model2.json', 'model3.json', 'model4.json', 'model5.json',
             'model6.json', 'model7.json', 'model8.json', 'model9.json', 'model10.json']
    
    mean = predictSeqs(embedTest, weights, jsonFiles)
    predict = ensembleModel(embedTest, mean)
    saveResults(Seq_id, predict)
    
if __name__ == '__main__':
    main()

