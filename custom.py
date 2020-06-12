import glob
import pickle
import numpy
import random
from music21 import *
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

import os
import numpy as np

def read_midi(file):
    #Function takes input midi file and returns notes in a list
    notes = []
    midi = converter.parse(file)
    print("Parsing %s" % file)
    notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    print(len(notes))
    
    return np.array(notes)

def read_data(path):
    #function to read all the midi files
    files=[i for i in os.listdir(path+'/') if i.endswith(".mid")]
    notes_array = np.array([read_midi(path+'/'+i) for i in files])
    notes_ = [element for note_ in notes_array for element in note_]
    #No. of unique chords in the dataset
    unique_notes = list(set(notes_))
    n_vocab = len(unique_notes)
    print("unique chords",len(unique_notes))
    #creating a dictionary for all chords
    notes_dictionary = dict((note_, number) for number, note_ in enumerate(unique_notes))
    
    return notes_,n_vocab,notes_dictionary

def prepare_data(notes_,notes_dictionary,lookback):
    transformed_notes = []
    for i in notes_:
        transformed_notes.append(notes_dictionary[i])        
    x = []
    y = []
    for i in range(0, len(notes_) - lookback-1):
        input_ = transformed_notes[i:i + lookback]
        output = transformed_notes[i + lookback]
        x.append(input_)
        y.append(output)
    x=np.array(x)
    x=x/float(len(notes_dictionary))
    y=np.array(y)
    x=np.reshape(x,(x.shape[0],lookback,1))
    
    return x,y

def get_transition_matrix(notes,n_vocab):
    trans = np.zeros((n_vocab,n_vocab))
    for i in range(n_vocab):
        for j in range(n_vocab):
            no_of_i = notes.count(i)
            no_of_j = notes.count(j)
            countij = 0
            countji = 0
            for k in range(len(notes)-1):
                if notes[k]==i and notes[k+1]== j:
                    countij+=1
                if notes[k]==j and notes[k+1] == i:
                    countji+=1
            trans[i,j]=countij/no_of_i
            trans[j,i]=countji/no_of_j
    
    return trans

def generate_stochastic_music(trans,start,pitch):
    
    int_to_note = dict((number, note) for number, note in enumerate(pitch))
    prediction_output = []
    
    # generate 50 elements
    for note_index in range(50):
    
        index = pick_chord_probs(trans[start])
        #convert integer back to the element
        pred = int_to_note[index]
        prediction_output.append(pred)
        start=index

    return prediction_output

def pick_chord_probs(arr):
    r = np.random.rand()
    cumprobs = np.cumsum(arr)
    for i in range(len(cumprobs)):
        if r<cumprobs[i]:
            return i

def generate_music(model, pitch, no_of_timesteps, pattern):
    
    int_to_note = dict((number, note) for number, note in enumerate(pitch))
    prediction_output = []
    
    # generating 20 notes
    for note_index in range(20):
        
        #reshaping array to feed into trained model
        input_ = np.reshape(pattern, (1, len(pattern),1))
        proba = model.predict(input_, verbose=0)[0]
        index = np.argmax(proba)
        
        #convert integer back to the chord element
        pred = int_to_note[index]
        prediction_output.append(pred)
        pattern = list(pattern)
        pattern.append(index/float(len(pitch)))
        
        #removing the chord at index 0
        pattern = pattern[1:len(pattern)]

    return prediction_output

def convert_to_midi(prediction_output,output_dir):
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        
        #setting duration for chords
        dur = 2
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.storedInstrument = instrument.Guitar
            new_chord.offset = offset
            new_chord.duration = duration.Duration(dur)
            output_notes.append(new_chord)
        offset += dur
    midi_stream = stream.Stream(output_notes)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    midi_stream.write('midi', fp=output_dir+'/output_music.mid')

def cnn(n_vocab,no_of_timesteps):
    K.clear_session()
    model = Sequential()
    model.add(Conv1D(32,3,activation='relu',padding='causal',input_shape=(no_of_timesteps,1)))
    model.add(MaxPool1D(2))
    model.add(Conv1D(64,3,activation='relu',padding='causal'))
    model.add(MaxPool1D(2))
    model.add(Conv1D(128,3,activation='relu',padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
    model.add(Conv1D(256,3,activation='relu',padding='causal'))
    model.add(Dropout(0.4))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics= ['accuracy'])
    model.summary()
    
    return model

def lstm(n_vocab,no_of_timesteps):
  K.clear_session()
  model = Sequential()
  model.add(LSTM(128,input_shape = (no_of_timesteps,1),return_sequences=True,activation='tanh'))
  model.add(Dropout(0.4))
  model.add(LSTM(128))
  model.add(Activation('tanh'))
  model.add(Dense(n_vocab))
  model.add(Activation('softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics= ['accuracy'])
  model.summary()

  return model