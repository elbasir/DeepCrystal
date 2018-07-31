import numpy as np
np.random.seed(1337)
from keras.models import load_model
import pandas as pd
import io
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dropout, Input, Convolution1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.layers import  merge
from keras.regularizers import l2
from keras.layers import AveragePooling1D
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from keras.models import model_from_json

def parseFastaFile():
    X_train = []
    y_train = []
    fasta_sequences = SeqIO.parse(open("train.fasta"),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, fasta.seq.tostring()
        X_train.append(sequence)
        if('Crystallizable' in name):
            y_train.append(1)
        else:
            y_train.append(0)
            
    return(X_train, y_train)

def onehotarray(X_train, y_train):
    for i in range(0, len(X_train)):
    train = []
    st = str(X_train[i])
    trainStr =''
    for ch in st:
        if(ch =='B' or ch =='J' or ch =='O' or ch =='U' or ch =='Z'):
            trainStr += 'X'
        else:
            trainStr += ch
        
    train.append(trainStr)
    X_train[i] = train
    
    amino_acids ='ACDEFGHIKLMNPQRSTVWXY'
    for i in range(0, len(X_train)):
        train = []
        st = str(X_train[i])
        trainStr =''
        for ch in st:
            if(ch in amino_acids):
                trainStr += ch
        
        train.append(trainStr)
        X_train[i] = train
    
    X = []
    Y = []
    for i in range(len(X_train)):
        if(len(X_train[i][0]) <= 800):
            X.append(X_train[i][0])
            Y.append(y_train[i])
        
    X_train = np.array(X)
    y_train = np.array(Y)
    X_train = X_train.reshape(len(X_train),1)
    return(X_train, y_train)
    
    
def embedding(X_train):
    embed = []
    for i in range(0, len(X_train)):
        length = len(X_train[i][0])
        pos = []
        counter = 0
        st = X_train[i][0]
        for c in st:
            AMINO_INDEX = amino_acids.index(c)
            pos.append(AMINO_INDEX)
            counter += 1
        while(counter < 800):
            pos.append(21)
            counter += 1
        embed.append(pos)
    embed = np.array(embed)
    return(embed)

def validate_train_data(Label, data):
    X_train1 = []
    X_validate = []
    y_train1 = []
    y_validate = []
    val_count = 0
    val0_count = 0
    val1_count = 0
    for i in range(0, len(Label)):
        if(Label[i] == 0 and val0_count <= 2066):
            val0_count += 1
            X_validate.append(data[i])
            y_validate.append(Label[i])
        elif(Label[i] == 0 and val0_count > 2066):
            X_train1.append(data[i])
            y_train1.append(Label[i])
        elif(Label[i] == 1 and val1_count <= 529):
            val1_count += 1
            X_validate.append(data[i])
            y_validate.append(Label[i])
        elif(Label[i] == 1 and val1_count > 529):
            X_train1.append(data[i])
            y_train1.append(Label[i])
        
        
    X_train1 = np.array(X_train1)
    X_validate = np.array(X_validate)
    y_train1 = np.array(y_train1)
    y_validate = np.array(y_validate)
    return(X_train1, X_validate, y_train1, y_validate)

def train_model(X_train1, X_validate, y_train1, y_validate):
    lr = 0.001
    pl = 5
    l2value = 0.001
    stride_ = 1
    stride_max = 1
    border = 'same'
    main_input = Input(shape=(800,), dtype='int32', name='main_input')

    x = Embedding(output_dim=50, input_dim=22, input_length=800)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(c)

    d = Convolution1D(64, 9, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    dpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(d)


    f = Convolution1D(64, 4, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    fpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(f)

    g = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    gpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(g)

    h = Convolution1D(64, 6, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    hpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(h)

    i = Convolution1D(64, 7, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    ipool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(i)


    merge2 = merge([apool, bpool, cpool, dpool,fpool,gpool,hpool, ipool], mode='concat', concat_axis=-1)
    merge2 = Dropout(0.3)(merge2)

    scalecnn1 = Convolution1D(64, 11, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
    scale1 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(scalecnn1)


    scalecnn2 = Convolution1D(64, 13, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
    scale2 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(scalecnn2)


    scalecnn3 = Convolution1D(64, 15, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
    scale3 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(scalecnn3)

    scale = merge([scale1, scale2, scale3], mode='concat', concat_axis=-1)

    scale = Dropout(0.3)(scale)

    cnn1 = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
    cnn10 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(cnn1)

    cnn2 = Convolution1D(64, 9, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
    cnn20 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(cnn2)

    cnn3 = Convolution1D(64, 13, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
    cnn30 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode=border)(cnn3)

    cnn50 = merge([cnn10, cnn20, cnn30], mode='concat', concat_axis=-1)

    cnn50 = Dropout(0.3)(cnn50)
    x = Flatten()(cnn50)

    x = Dense(256, activation='relu', name='FC', W_regularizer=l2(l2value))(x)

    output = Dense(1,activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)
    model = Model(input=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    best_Weight_File="name of the weight file.hdf5"
    checkpoint = ModelCheckpoint(best_Weight_File, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]


    model.fit(X_train1, y_train1, validation_data=(X_validate, y_validate), class_weight=class_weight_dict, nb_epoch=300, batch_size=64, callbacks=callback_list)
    
    return(model)

def save_model(model):
    model_json = model.to_json()
    with open("name of the json file.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("name of the model file.h5")
    print("Model is saved")
    
def main():
    
    X_train = []
    y_train = []
    X_train, y_train = parseFastaFile()
    X_train, y_train = onehotarray(X_train, y_train):

    
    embed = embedding(X_train) 
    data,Label = shuffle(embed,y_train, random_state=2)
    X_train1, X_validate, y_train1, y_validate = validate_train_data(Label, data)
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train1), y_train1)
    class_weight_dict = dict(enumerate(class_weight))
    model = train_model(X_train1, X_validate, y_train1, y_validate)
    save_model(model)
    
if __name__ == '__main__':
    main()

