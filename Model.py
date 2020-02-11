import numpy as np
np.random.seed(1337)
import pandas as pd
import io
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Convolution1D, MaxPooling1D, Flatten, merge, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle, class_weight
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab
import itertools


lr = 0.001 # Learning rate
pl = 5 
l2value = 0.001 # L2 regularization value
stride_ = 1
stride_max = 1
#border = 'same'

main_input = Input(shape=(800,), dtype='int32', name='main_input')
x = Embedding(output_dim=50, input_dim=22, input_length=800)(main_input)
a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
apool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(a)
b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
bpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(b)
c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
cpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(c)
d = Convolution1D(64, 9, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
dpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(d)
f = Convolution1D(64, 4, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
fpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(f)
g = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
gpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(g)
h = Convolution1D(64, 6, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
hpool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(h)
i = Convolution1D(64, 7, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
ipool = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(i)
merge2 = merge([apool, bpool, cpool, dpool,fpool,gpool,hpool, ipool], mode='concat', concat_axis=-1)
merge2 = Dropout(0.3)(merge2)
scalecnn1 = Convolution1D(64, 11, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
scale1 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(scalecnn1)
scalecnn2 = Convolution1D(64, 13, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
scale2 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(scalecnn2)
scalecnn3 = Convolution1D(64, 15, activation='relu', border_mode='same', W_regularizer=l2(l2value))(merge2)
scale3 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(scalecnn3)
scale = merge([scale1, scale2, scale3], mode='concat', concat_axis=-1)
scale = Dropout(0.3)(scale)
cnn1 = Convolution1D(64, 5, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
cnn10 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(cnn1)
cnn2 = Convolution1D(64, 9, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
cnn20 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(cnn2)
cnn3 = Convolution1D(64, 13, activation='relu', border_mode='same', W_regularizer=l2(l2value))(scale)
cnn30 = MaxPooling1D(pool_length=pl, stride=stride_max, border_mode='same')(cnn3)
cnn50 = merge([cnn10, cnn20, cnn30], mode='concat', concat_axis=-1)
cnn50 = Dropout(0.3)(cnn50)
x = Flatten()(cnn50)
x = Dense(256, activation='relu', name='FC', W_regularizer=l2(l2value))(x)
output = Dense(1,activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)
model = Model(input=main_input, output=output)
adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
best_Weight_File="/name_of_the_weight_File.hdf5"
checkpoint = ModelCheckpoint(best_Weight_File, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(X_train1, y_train1, validation_data=(X_validate, y_validate), class_weight=class_weight_dict, nb_epoch=300, batch_size=64, callbacks=callback_list)

# Saving json and model files

model_json = model.to_json()
with open("/name_of_json_file.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("/name_of_model.h5")
print("Saved model to disk")

