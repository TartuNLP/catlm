#!/usr/bin/env python

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
		
		data, params = txt.loadAndClean(dataFile, maxLen, chars = doChars)
		
		print(data.txtIn[0])
		print(data.catIn[0][0])
		print(data.catIn[1][0])
		
		rnnlm.saveParams(params, paramOutFile)
		
		lm = rnnlm.initModelNew(params, embSize = (32 if doChars else 256))
		
		rnnlm.learn(lm, data)
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
		
		rnnlm.learn(lm, data)
		lm.save(modelOutFile)
	else:
		raise Exception("AAAAA")
