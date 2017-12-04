#!/usr/bin/env python

import sys
import rnnlm
import txt
import pickle
from keras.models import load_model

if __name__ == "__main__":
	modelInFile = sys.argv[1]
	dictInFile = sys.argv[2]
	try:
		catSpec = sys.argv[3]
	except IndexError:
		catSpec = ""

	numToSample = 1
	
	(mdl, params) = rnnlm.loadModels(modelInFile, dictInFile)
	
	for _ in range(numToSample):
		raw, prob = rnnlm.sample(mdl,  params, txt.spec2vec(params, catSpec))
		
		decoded = [str(params.i2w[i]) for i in raw]
		print("".join(decoded).encode(encoding='utf8') + " (" + str(prob) + ")")
