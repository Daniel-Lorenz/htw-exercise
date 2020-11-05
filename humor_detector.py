#!/usr/bin/env python
# coding: utf-8
import logging
import gzip
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class HumorDetector:

    def __init__(self) -> None:
        self.embeddings_index = {}
        self.__build_embedding_index__()
        self.logging = self.__prepare_logger__()
        self.max_number_of_words_in_sentence = 50
        self.vocabulary_size = 10000
        self.labels, self.texts = self.__load_labels_and_texts__()
        self.data, self.embedding_dim, self.embedding_matrix, self.labels = self.__preprocess_data__(self.labels,
                                                                                                     self.texts)
        self.logging.info("This is your HumorDetector. I am ready to work!")

    def __build_embedding_index__(self):
        filename =["glove.6B/glove.6B.100d.split.0.txt", "glove.6B/glove.6B.100d.split.1.txt", "glove.6B/glove.6B.100d.split.2.txt",
                   "glove.6B/glove.6B.100d.split.3.txt"]
        content = ""
        for i in range(len(filename)):
            with open(filename[i], encoding="utf8") as file:
                content = content+file.read()
        content = content.split("\n")
        content = content[0:-1]
        for line in content:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs

    def __prepare_logger__(self):
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger

    def train_model_and_plot_results(self, model):
        self.__train_and_plot_results__(self.data, self.labels, model)

    def __train_and_plot_results__(self, data, labels, model):
        history = self.__compile_and_fit_model__(data, labels, model)
        self.plot_result(history)

    def __compile_and_fit_model__(self, data, labels, model):
        self.logging.info("I will start model compilation.")
        model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', self.__f1__])
        self.logging.info("Model has been compiled.")
        self.logging.info("I will start training.")
        history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
        self.logging.info("Model has been trained.")
        return history

    @staticmethod
    def plot_result(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        f1 = history.history['__f1__']
        val_f1 = history.history['val___f1__']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

        plt.figure()

        plt.plot(epochs, f1, label='Training fmeasure')
        plt.plot(epochs, val_f1, label='Validation fmeasure')
        plt.title('Training and validation fmeasure')
        plt.xlabel('epochs')
        plt.ylabel('f1')
        plt.legend()

        plt.show()

    def __preprocess_data__(self, labels, texts):
        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen=self.max_number_of_words_in_sentence)
        labels = np.array(labels)
        # shuffle the data
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        # parsing the GloVe word-embeddings file
        embeddings_index = self.embeddings_index
        # preparing glove word embeddings matrix
        embedding_dim = 100
        embedding_matrix = np.zeros((self.vocabulary_size, embedding_dim))
        for word, i in word_index.items():
            if i < self.vocabulary_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector  # for words not in embedding index values will be zeros
        return data, embedding_dim, embedding_matrix, labels

    def __load_labels_and_texts__(self):
        # Read in the lists of sentences from respective pickle files
        humour, proverb, reuters, wiki = self.__load_data__()
        texts = []
        labels = []
        # shuffling the different negative samples
        neg = proverb + wiki + reuters
        np.random.shuffle(neg)
        # adding the positive samples
        for line in humour:
            texts.append(line)
            labels.append(1)
        # taking equal samples from both classes
        neg = neg[:len(humour)]
        # adding the negative samples
        for line in neg:
            texts.append(line)
            labels.append(0)
        return labels, texts

    def __load_data__(self):
        humour = pd.read_pickle('datasets/humorous_oneliners_win.pickle')
        proverb = pd.read_pickle('datasets/proverbs_win.pickle')
        wiki = pd.read_pickle('datasets/wiki_sentences_win.pickle')
        reuters = pd.read_pickle('datasets/reuters_headlines_win.pickle')
        return humour, proverb, reuters, wiki

    # to compute fmeasure as custom metric
    def __f1__(self, y_true, y_pred):
        precision = self.__precision__(y_true, y_pred)
        recall = self.__recall__(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def __recall__(self, y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def __precision__(self, y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
