import tensorflow as tf
import torch
from konlpy.tag import Okt

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import random
import time
import datetime
import re
import pickle
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

global a 


# 파일로부터 모델을 읽는다. 없으면 생성한다.
try:
    print("ok")
    data = pd.read_csv('KorCCViD_v1.3_fullcleansed.csv')
    print("ok")
    #train data & test data 로드
    train_data = data[:900]
    test_data = data[900:]
    print("ok")
    okt = Okt()
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    X_train = []
    for sentence in tqdm(train_data['Transcript']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_train.append(stopwords_removed_sentence)
    X_test = []
    for sentence in tqdm(test_data['Transcript']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_test.append(stopwords_removed_sentence)
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    
# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
        
    vocab_size = total_cnt - rare_cnt + 1

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['Label'])
    y_test = np.array(test_data['Label'])
    max_len = 200

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)



    embedding_dim = 100
    hidden_units = 128
    l2_lambda = 0.02  # L2 규제 강도 조절값

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units, kernel_regularizer=l2(l2_lambda)))  # L2 규제 적용
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=7, callbacks=[es, mc], batch_size=64, validation_split=0.2)




except:
    print('gd')

def faq_answer(new_sentence):
    a=0
    loaded_model = load_model('best_model.h5')
     # new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 200) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    
    if(score > 0.5):
        a+=2
        return("{:.2f}% 확률로 보이스피싱입니다.\n".format(score * 100+a))
    else:
        a+=2
        return("{:.2f}% 확률로 보이스피싱입니다.\n".format((1 - score) * 100+a))


