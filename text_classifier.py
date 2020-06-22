import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sn
from tensorflow import keras
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('datasets/stack_overflow_questions.csv')

# Removing stop words from the question titles
stop = stopwords.words('english')
data['title_without_stopwords'] = data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Inspect results
data

titles = data['title_without_stopwords'].str.lower()
labels = data['label']

vocab_size = 10000
embedding_dim = 20
max_length = 80
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = round(data.shape[0]*0.8)

training_titles = titles[0:training_size]
testing_titles = titles[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_titles)

word_index = tokenizer.word_index

training_titles = tokenizer.texts_to_sequences(training_titles)
training_padded = pad_sequences(training_titles,
                                maxlen = max_length,
                                padding = padding_type,
                                truncating = trunc_type)

testing_titles = tokenizer.texts_to_sequences(testing_titles)
testing_padded = pad_sequences(testing_titles,
                               maxlen = max_length,
                               padding = padding_type,
                               truncating = trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

num_epochs = 50
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

model.evaluate(testing_padded, testing_labels)

pred_class = model.predict(testing_padded)
obsv_class = testing_labels

res = pd.DataFrame()
res['Observado'] = obsv_class
res['Predito'] = pred_class
res = res.round()

cm = confusion_matrix(res['Observado'], res['Predito'])

confusion_matrix = pd.crosstab(res['Observado'], res['Predito'], rownames=['Observado'], colnames=['Predito'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()
