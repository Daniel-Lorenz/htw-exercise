import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, Dense, GRU, Conv1D, LSTM, MaxPooling1D, SimpleRNN, Flatten, Dropout
from humor_detector import HumorDetector

detector = HumorDetector(student="Daniel Lorenz")
embedding_dim = detector.embedding_dim
embedding_matrix = detector.embedding_matrix
vocabulary_size = detector.vocabulary_size
words_in_sentence = detector.max_number_of_words_in_sentence

model = Sequential()
# TODO Start thinking/coding here

model.add(Embedding(vocabulary_size, 32))

# TODO End thinking/coding here
model.add(Dense(1, activation='sigmoid'))
model.summary()

detector.train_model_and_plot_results(model)
