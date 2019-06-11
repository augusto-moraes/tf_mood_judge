import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = Tokenizer(num_words=3000)

labels = ['negativo', 'positivo']

with open('out/twtdictionary.json', 'r') as dictionary_file:
  dictionary = json.load(dictionary_file)
  
def convert_text_to_index_array(text):
  words = kpt.text_to_word_sequence(text)
  wordIndices = []
  for word in words:
    if word in dictionary:
      wordIndices.append(dictionary[word])
    else:
      print("'%s' not in training corpus; ignoring." %(word))
  return wordIndices

json_file = open('out/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

model.load_weights('out/model.h5')

def avaliar(evalSentence):
  if len(evalSentence) == 0:
    return
  
  testArr = convert_text_to_index_array(evalSentence)
  input = tokenizer.sequences_to_matrix([testArr], mode='binary')
  
  pred = model.predict(input)
  
  print("Sentimento %s; %f%% de confianca!!" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)]))
  

print("Insira a frase a ser avaliada:")
print("**Digite \q para sair**")
quote = input()
while quote != "\q":
	avaliar(quote)
	quote = input()
