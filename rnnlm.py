from __future__ import print_function

import sys
import numpy as np
import math
import pickle

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import to_categorical

from keras.models import load_model

from txt import SOS, EOS, OOV

def initModelNew(params, embSize = 512, hdnSize = 1024, catEmbSize = 8):
	# main input
	inputs = [Input(shape=(params.max, len(params.w2i)))]
	
	# inputs for each additional cat
	for i, cat2idx in enumerate(params.c2i):
		inputs.append(Input(shape=(params.max, len(cat2idx))))
	
	# feed-forward embeddings for each input separately
	embeddings = [Dense(embSize if i == 0 else catEmbSize, activation='linear')(inLayer) for i, inLayer in enumerate(inputs)]

	embConc = concatenate(embeddings)
	
	hidRec1 = Dropout(0.2)(LSTM(hdnSize, return_sequences=True)(embConc))
	
	hidRec2 = Dropout(0.2)(LSTM(hdnSize, return_sequences=True)(hidRec1))
	
	output = Dense(len(params.w2i), activation='softmax')(hidRec2)
	
	model = Model(inputs=inputs, outputs=[output])
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
	
	return model

#def initModelOld(vocSize, maxLen, embSize = 512, hdnSize = 1024):
#	model = Sequential()
#
#	model.add(Embedding(input_dim = vocSize, output_dim = embSize, input_length = maxLen))
#
#	model.add(LSTM(hdnSize, input_shape=(maxLen, embSize), return_sequences=True))
#	model.add(Dropout(0.2))
#
#	model.add(LSTM(hdnSize, input_shape=(maxLen, hdnSize), return_sequences=True))
#	model.add(Dropout(0.2))
#
#	model.add(Dense(vocSize))
#	model.add(Activation('softmax'))
#
#	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
#
#	return model

def learn(mdl, data):
	mdl.fit([data.txtIn] + data.catIn, data.out, epochs=1, batch_size=40)

def renorm(pd, temp = 0.5):
	raw = [p**(1/temp) for p in pd]
	raw[OOV] = 0
	s = sum(raw)
	return [p/s for p in raw]

def sample(mdls, catVecs, temp = 1.0):
	(mdl, dicts) = mdls
	
	baseInput = np.zeros([1, dicts['m']], dtype='int32')
	
	result = []
	w = SOS
	
	prob = 0.0
	
	for i in range(dicts['m']):
		baseInput[0, i] = w
		
		pd = mdl.predict(baseInput)[0, i]
		
		#w = max(enumerate(pd), key=lambda x: x[1] if x[0] != OOV else 0)[0]
		w = np.random.choice(dicts['v'], p = renorm(pd, temp))
		prob += math.log(pd[w])
		
		if w == EOS:
			break
		
		result.append(w)
	
	return result, prob/(len(result)+1)

def score(snt, models, catVecs, skipEOS = False):
	(mdl, dicts) = models
	
	inputs, outputs = text2numio([snt], dicts['w2i'], dicts['m'])
	
	#print(inputs)
	
	hyps = mdl.predict(inputs)
	
	result = 0
	length = 0
	
	for j, pVec in enumerate(hyps[0]):
		inp = inputs[0, j]
		outp = outputs[0, j, 0]
		
		if inp == 0 or (skipEOS and outp == EOS):
			break
		
		#print(j, outp, pVec)
		
		length += 1
		result += math.log(pVec[outp])
		
	return result / length

def loadModels(modelFile, paramFile):
	mdl = load_model(modelFile)
	
	with open(paramFile, 'rb') as fh:
		params = pickle.load(fh)
	
	return (mdl, params)

def saveParams(metaparams, filename):
	with open(filename, 'wb') as fh:
		pickle.dump(metaparams, fh, protocol=pickle.HIGHEST_PROTOCOL)
