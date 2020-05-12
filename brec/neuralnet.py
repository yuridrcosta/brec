import tensorflow as tf
import pandas as pd
import numpy as np
import json
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from django.conf import settings
from breclib import settings
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def create_model( vocabulary_size,num_classes, max_length, units=100, dense_neurons=16, embedding_vector_length=300):
    input = Input(shape=(max_length,))
    model = Embedding(input_dim= vocabulary_size+1, output_dim=embedding_vector_length, input_length=max_length)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    #model = TimeDistributed(Dense(dense_neurons, activation='relu'))(model)
    #model = Dense(num_classes, activation="softmax")(model)
    out = TimeDistributed(Dense(3, activation="softmax"))(model)
    #crf = CRF(3, name="output")
    #out = crf(model)
  # softmax output layer
    model = Model(input,out)
    #model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy],sample_weight_mode="temporal")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'],sample_weight_mode="temporal")
    return model

def evaluate_text(model,x_sample,sentences_text):
    evaluation = []
    for sentence_pos in range(len(sentences_text)):
        sentence_evaluation = []
        p = []
        p = model.predict(np.array([x_sample[sentence_pos]]))
        p = np.argmax(p,axis=-1)
        for word_pos in range(len(sentences_text[sentence_pos])):
            sentence_evaluation.append((sentences_text[sentence_pos][word_pos],p[0][word_pos])) # Estou adicionando a lista uma tupla com a palavra e a classificação
        evaluation.append(sentence_evaluation)
    return evaluation




def run_evaluation(text):
    #print('print 2:{}'.format(text))
    max_len = 30
    #with open('/Home/Desktop/heroku/breclib/brec/static/word_index.json', 'r', encoding='utf-8') as f:
    with open(os.path.join(THIS_FILE_PATH, 'static/word_index.json'), 'r', encoding='utf-8') as f:
        word2idx = json.load(f)

    model = create_model(len(word2idx),3,30)   
    #model.load_weights('/Home/Desktop/heroku/breclib/brec/static/model-newtokenizing2-96-75.h5')
    model.load_weights(os.path.join(THIS_FILE_PATH, 'static/model-newtokenizing2-96-75.h5'))

    splitted_text = text.split(' ')
    #print(splitted_text)

    count = 0
    sentences_text = []
    sent_text = []
    text_size = len(splitted_text)
    for i in splitted_text:
        sent_text.append(i)
        count+=1
        if count == max_len:
            sentences_text.append(sent_text)
            sent_text = []
            text_size -=30
            count = 0
        if(count == text_size):
            sentences_text.append(sent_text)
    #print(sentences_text)  

    X_sample = []
    for s in sentences_text:
        aux = []
        for w in s:
            if w in word2idx:
                aux.append(word2idx[w])
            else:
                aux.append(word2idx['UNK'])
        X_sample.append(aux) 
    #print(X_sample)


    x_sample = pad_sequences(sequences = X_sample, maxlen=max_len,value=word2idx["PAD"], padding='post')
    #print(x_sample)
    #print(r_tokenized)
    #print(len(r_tokenized))
    #print("Word -> Tag")
    #for i in range(len(sentences_text)):
    #predict_sentence(x_sample[i],i)
    text_evaluated = evaluate_text(model,x_sample,sentences_text)
    #print(text_evaluated)
    return text_evaluated
