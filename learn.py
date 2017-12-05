#!/usr/bin/env python3

import sys
import rnnlm
import txt
import pickle
from keras.models import load_model

if __name__ == "__main__":
	if len(sys.argv) == 4:
		#learn from scratch
		dataFile = sys.argv[1]
		paramOutFile = sys.argv[2]
		modelOutFile = sys.argv[3]
		
		maxLen = 50
		doChars = True
		
		txtdata, params = txt.loadAndClean(dataFile, maxLen, chars = doChars)
		
		rnnlm.saveParams(params, paramOutFile)
		
		lm = rnnlm.initModelNew(params, embSize = (32 if doChars else 256))
		
		rnnlm.learn(lm, params, txtdata)
		lm.save(modelOutFile)
		
	elif len(sys.argv) == 5:
		#continue learning
		dataFile = sys.argv[1]
		paramInFile = sys.argv[2]
		modelInFile = sys.argv[3]
		modelOutFile = sys.argv[4]
		
		(lm, params) = rnnlm.loadModels(modelInFile, paramInFile)
		
		txtData, _, _ = txt.loadFile(dataFile, maxLen = params.max, chars = True)
		data = txt.getIOData(txtData, params)
		
		rnnlm.learn(lm, params, data)
		lm.save(modelOutFile)
	else:
		raise Exception("AAAAA")
