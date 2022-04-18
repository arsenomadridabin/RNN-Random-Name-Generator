from numpy.lib.npyio import load
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import os
from keras.models import load_model


epochs = 5 # Number of times we train on our full data
batch_size = 32 # Data samples in each training step
dropout_rate = 0.2 # Regularization with dropout
num_chars = 27
step_length = 1 

no_of_names = 5

model_link = "https://drive.google.com/file/d/1G3Vx5GTTefpf27-zfDm6s2b0wvgX0rTR/view?usp=sharing"

def is_not_speical_character_in_name(name):
  flag = True
  for char in name:
    if ord(char) < 97 or ord (char) > 122:
      flag = False
      return False
  return True


def create_model(max_len):
  model = Sequential()
  model.add(LSTM(128,
                input_shape=(max_len, num_chars),
                recurrent_dropout=dropout_rate))
  model.add(Dense(units=num_chars, activation='softmax'))

  optimizer = RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,metrics=['acc'])
  return model


def train_model(model):
  with open('names.txt') as f:
      records = f.readlines()

  #Get only the names from each line
  input_names = [each.split(',')[0].lower() for each in records]

  input_names = list(filter(is_not_speical_character_in_name, input_names))


  # Make it all to a long string
  concat_names = '\n'.join(input_names).lower()

  # Find all unique characters by using set()
  chars = sorted(list(set(concat_names)))
  num_chars = len(chars) -1 + 2

  # Build translation dictionaries, 'a' -> 0, 0 -> 'a'
  char2idx = dict((c, i) for i, c in enumerate(chars))
  idx2char = dict((i, c) for i, c in enumerate(chars))

  # Use longest name length as our sequence window
  max_sequence_length = max([len(name) for name in input_names])

  print('Total chars: {}'.format(num_chars))
  print('Corpus length:', len(concat_names))
  print('Number of names: ', len(input_names))
  print('Longest name: ', max_sequence_length)


  # Make it all to a long string
  concat_names = '\n'.join(input_names).lower()

  # Find all unique characters by using set()
  chars = sorted(list(set(concat_names)))
  # -1 fpr \n and +2 fpr begining char (^) and end charater ($)
  num_chars = 27

  # Build translation dictionaries, 'a' -> 0, 0 -> 'a'
  char2idx = dict((c, i) for i, c in enumerate(chars))
  idx2char = dict((i, c) for i, c in enumerate(chars))

  # Use longest name length as our sequence window
  max_sequence_length = max([len(name) for name in input_names])

  print('Total chars: {}'.format(num_chars))
  print('Corpus length:', len(concat_names))
  print('Number of names: ', len(input_names))
  print('Longest name: ', max_sequence_length)

  sequences = []
  next_chars = []

  # Loop over our data and extract pairs of sequances and next chars
  for i in range(0, len(concat_names) - max_sequence_length, step_length):
      sequences.append(concat_names[i: i + max_sequence_length])
      next_chars.append(concat_names[i + max_sequence_length])

  num_sequences = len(sequences)

  print('Number of sequences:', num_sequences)
  print('First 10 sequences and next chars:')
  for i in range(10):
      print('X=[{}] y=[{}]'.replace('\n', ' ').format(sequences[i], next_chars[i]).replace('\n', ' '))


  X = np.zeros((num_sequences, max_sequence_length, num_chars), dtype=np.bool)
  Y = np.zeros((num_sequences, num_chars), dtype=np.bool)

  for i, sequence in enumerate(sequences):
      for j, char in enumerate(sequence):
          X[i, j, char2idx[char]] = 1
          Y[i, char2idx[next_chars[i]]] = 1

  print('X shape: {}'.format(X.shape))
  print('Y shape: {}'.format(Y.shape))

  history = model.fit(X,Y,epochs=5,batch_size=128,verbose=1)
  model.save('model_link.h5')

# model = create_model(15)

# train = train_model(model)

def generate_name(length):
  model = load_model('model_link.h5')

  with open('names.txt') as f:
    records = f.readlines()

  #Get only the names from each line
  input_names = [each.split(',')[0].lower() for each in records]

  input_names = list(filter(is_not_speical_character_in_name, input_names))


  # Make it all to a long string
  concat_names = '\n'.join(input_names).lower()

  # Start sequence generation from end of the input sequence
  sequence = concat_names[-(max_sequence_length - 1):] + '\n'

  new_names = []
  print('{} new names are being generated'.format(no_of_names))

  while len(new_names) < no_of_names:
      # Vectorize sequence for prediction
      x = np.zeros((1, max_sequence_length, num_chars))
      for i, char in enumerate(sequence):
          x[0, i, char2idx[char]] = 1

      # Sample next char from predicted probabilities
      probs = model.predict(x, verbose=0)[0]
      probs /= probs.sum()
      next_idx = np.random.choice(len(probs), p=probs)
      next_char = idx2char[next_idx]
      sequence = sequence[1:] + next_char

      # New line means we have a new name
      if next_char == '\n':
          gen_name = [name for name in sequence.split('\n')][1]
          
          if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
              gen_name = gen_name[1:]
          
          # Discard all names that are too short
          if len(gen_name) > 2:
              # Only allow new and unique names
              if gen_name not in input_names + new_names:
                  if len(gen_name) == length:
                    new_names.append(gen_name)
          
          if 0 == (len(new_names) % (no_of_names/ 10)):
              pass


  print_first_n = min(10, no_of_names)
  print('First {} generated names:'.format(print_first_n))
  for name in new_names[:print_first_n]:
      print(name)



if __name__ == "__main__":

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)

  model = create_model(15)
  train_model(model)
  generate_name(6)
