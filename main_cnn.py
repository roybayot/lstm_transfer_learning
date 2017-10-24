#!/usr/bin/env python2
# coding=utf-8

from __future__ import print_function
import numpy as np
import gensim

parser = argparse.ArgumentParser()
parser.add_argument("--seed=",  
                     dest="SEED", 
                     default=1337)
args = parser.parse_args()
np.random.seed(args.SEED)

import sys
import argparse
import keras
import pandas as pd

from datetime import datetime
from scipy.stats import mode
from sklearn import preprocessing

from utils import make_X, make_y, make_train_val_split, make_lower_case, create_model_cnn
from config import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv1D, Embedding, Merge, Dropout
from keras.layers.core import Lambda
from keras.models import Model
from keras.constraints import maxnorm
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

train_csv_file = pd.read_csv(TRAIN_DATASET, delimiter="\t")
train_csv_file = train_csv_file.dropna()

# make the X
input_text = train_csv_file['text']
X = make_X(input_text=input_text, tokenizer=tokenizer, MAX_SEQUENCE_LENGTH)

# make the y
y = train_csv_file[TASK] 
y = make_y(y)

x_train, y_train, x_val, y_val = make_train_val_split(X, y, VALIDATION_SPLIT)

embeddings_index = load_word_vectors(W2V_FILE)

word_index = tokenizer.word_index
num_output_units =  y_train.shape[1]

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = make_embedding_matrix(word_index, embeddings_index, nb_words, EMBEDDING_DIM)

model = create_model_cnn()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['acc'])


time_now = str(datetime.now())
csv_filename = OUT_DIR + time_now + " csv_log-seed-{}.txt".format(SEED)

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1), 
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001),
             CSVLogger(csv_filename)
        	]

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=N_EPOCHS,
        shuffle=True, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

if EVAL_TEST:
	test_csv_file = pd.read_csv(TEST_DATASET, delimiter="\t")
    test_csv_file = test_csv_file.dropna()

    test_text = make_lower_case(test_csv_file['text'])
    test_sequences = tokenizer.texts_to_sequences(test_text)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test_y = make_y(test_text[TASK])

	preds = model.predict(test_data)
    score = model.evaluate(test_data, transformed_labels_test)

    category_predictions = np.argmax(preds, axis=1)
    test_csv_file["predictions"] = category_predictions
	
	test_filename = TEST_DATASET.split("/")[1] 
    test_csv_file_out_filename =  OUT_DIR + time_now +" predictions-per-tweet-" + test_filename + "-seed-{}".format(SEED)    
	test_csv_file.to_csv(test_csv_file_out_filename)


	users = list(set(test_csv_file["filename"]))

    correctly_classified = 0.0

    for user in users:
        predictions_per_user = test_csv_file[test_csv_file["filename"] == user]["predictions"]
        predicted_mode = mode(predictions_per_user)
        
        actual_class = test_csv_file[test_csv_file["filename"] == user][TASK]
        actual_mode = mode(actual_class)

        if predicted_mode[0][0] == actual_mode[0][0]:
            correctly_classified = correctly_classified + 1.0

    accuracy_over_users = correctly_classified / len(users)
    print("Accuracy over users: %f" % accuracy_over_users)
    
	text_to_write = "Training file: {}\nTest file:{}\nAccuracy:{}".format(TRAIN_DATASET, TEST_DATASET, accuracy_over_users)    
    
    out_filename_performance = OUT_DIR + time_now + " accuracy-over-users.txt"
    with open(out_filename_performance, 'w') as f:
    	f.write(text_to_write)

