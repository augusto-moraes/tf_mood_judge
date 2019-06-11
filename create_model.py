import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np

training = np.genfromtxt('static/data_pt.csv', delimiter=',', skip_header=1, usecols=(1,3), dtype=None, encoding='utf-8')

train_x = [x[1] for x in training]
train_y = np.asarray([x[0] for x in training])
print(train_y)

max_words = 3000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index

with open('out/twtdictionary.json', 'w') as dictionary_file:
  json.dump(dictionary, dictionary_file)
  
def convert_text_to_index_array(text):
  return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices=[]

for text in train_x:
  wordIndices = convert_text_to_index_array(text)
  allWordIndices.append(wordIndices)

allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
train_y = keras.utils.to_categorical(train_y, 2)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

ep = int(input("Quantidade de epocas: "))
if ep <= 0:
	while ep <= 0:
		print("Quantidade invalida, tente novamente !!\n")
		ep = int(input("Quantidade de epocas: "))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=ep, verbose=1, validation_split=0.1, shuffle=True)

model_json = model.to_json()
with open('out/model.json', 'w') as json_file:
  json_file.write(model_json)

model.save_weights('out/model.h5')

print('Saved Model ! :D')
