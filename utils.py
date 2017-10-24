#!/usr/bin/env python2

import pandas as pd
import numpy as np
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

from config import *

def make_lower_case(texts):
	lower_case_clean_train_data = []
    for each_line in texts:
        try:
            lower_case_clean_train_data.append(each_line.lower())
        except:
            lower_case_clean_train_data.append('')

    return lower_case_clean_train_data


def make_X(input_text, tokenizer, MAX_SEQUENCE_LENGTH)
	texts =  make_lower_case(input_text)
		if new_tokenizer == True:
			tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)
		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	return data

def make_y():
	labels = to_categorical(np.asarray(labels))
	return labels

def make_train_val_split(X, y, VALIDATION_SPLIT):
	indices = np.arange(X.shape[0])
	np.random.shuffle(indices)

	data = X[indices]
	labels = X[indices]

	nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

	x_train = data[:-nb_validation_samples]
	y_train = labels[:-nb_validation_samples]
	x_val = data[-nb_validation_samples:]
	y_val = labels[-nb_validation_samples:]

	return x_train, y_train, x_val, y_val



def load_word_vectors(W2V_FILE):
	try:
    	embeddings_index = gensim.models.word2vec.Word2Vec.load(W2V_FILE)
	except Exception as e:
    	print(e)
    else:
    	return embeddings_index


# max over time function
def max_1d(X):
    return K.max(X, axis=1)
 
# making the embedding matrix
def make_embedding_matrix(word_index, embeddings_index, nb_words, EMBEDDING_DIM):
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    index2word_set = set(embeddings_index.wv.index2word)

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        #embedding_vector = np.zeros(EMBEDDING_DIM)
        embedding_vector = np.random.uniform(-0.01,0.01,EMBEDDING_DIM)
        if word in index2word_set:
            embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector


    return embedding_matrix

def create_model_cnn():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')
	embedding_layer = Embedding(input_dim=nb_words + 1,
    	                        output_dim=EMBEDDING_DIM,
        	                    weights=[embedding_matrix],
            	                input_length=MAX_SEQUENCE_LENGTH,
                	            trainable=VEC_TRAINABLE)

	path_1_conv1d        = Conv1D(N_FEATURES, 5, activation=ACT_TYPE)
	path_1_max_over_time = Lambda(max_1d, output_shape=(N_FEATURES,))

	path_2_conv1d        = Conv1D(N_FEATURES, 4, activation=ACT_TYPE)
	path_2_max_over_time = Lambda(max_1d, output_shape=(N_FEATURES,))

	path_3_conv1d        = Conv1D(N_FEATURES, 3, activation=ACT_TYPE)
	path_3_max_over_time = Lambda(max_1d, output_shape=(N_FEATURES,))

	dropout_layer = Dropout(DROPOUT_VAL) 

	out_layer_1 = Dense(num_output_units, activation='softmax', kernel_constraint=maxnorm(W_CONST_MAXN)) 
	out_layer_2 = Dense(num_output_units, activation='softmax')
	out_layer_3 = Dense(num_output_units, activation='softmax', kernel_regularizer=l2(W_L2_REG))
	out_layer_4 = Dense(num_output_units, activation='softmax', kernel_regularizer=l2(W_L2_REG), kernel_constraint=maxnorm(W_CONST_MAXN))	


    embedding_sequence = embedding_layer(sequence_input)

    #% try new method of writing layers
    #%%==============================================================
    path_1 = path_1_conv1d(embedding_sequence)
    path_1 = path_1_max_over_time(path_1)

    #%%
    path_2 = path_2_conv1d(embedding_sequence)
    path_2 = path_2_max_over_time(path_2)


    #%%
    path_3 = path_3_conv1d(embedding_sequence)
    path_3 = path_3_max_over_time(path_3)


    #%%

    merged_vector = keras.layers.concatenate([path_1, path_2, path_3], axis=-1)

    out_merged = dropout_layer(merged_vector)


    if ARCH_NUM == 1:
        predictions = out_layer_1(out_merged) 
    elif ARCH_NUM == 2:
        predictions = out_layer_2(out_merged) 
    elif ARCH_NUM == 3:
        predictions = out_layer_3(out_merged)
    elif ARCH_NUM == 4:
        predictions = out_layer_4(out_merged)

    #%%==============================================================
    model = Model(sequence_input, predictions)
    return model