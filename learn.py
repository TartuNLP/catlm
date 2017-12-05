#!/usr/bin/env python3

import sys
import rnnlm
import txt
import pickle
from keras.models import load_model
from datetime import datetime

def log(msg):
	print("LOG", str(datetime.now()), msg)

if __name__ == "__main__":
	if len(sys.argv) == 4:
		#learn from scratch
		dataFile = sys.argv[1]
		paramOutFile = sys.argv[2]
		modelOutFile = sys.argv[3]
		
		maxLen = 50
		doChars = True
		
		log("load data")
		txtdata, params = txt.loadAndClean(dataFile, maxLen, chars = doChars)
		
		log("save params")
		rnnlm.saveParams(params, paramOutFile)
		
		log("init model")
		lm = rnnlm.initModelNew(params, embSize = (32 if doChars else 256))
		
		log("learn model")
		rnnlm.learn(lm, params, txtdata)
		
		log("save model")
		lm.save(modelOutFile)
		
		log("done")
		
	elif len(sys.argv) == 5:
		#continue learning
		dataFile = sys.argv[1]
		paramInFile = sys.argv[2]
		modelInFile = sys.argv[3]
		modelOutFile = sys.argv[4]
		
		log("load model and params")
		(lm, params) = rnnlm.loadModels(modelInFile, paramInFile)
		
		log("load data")
		txtData, _, _ = txt.loadFile(dataFile, maxLen = params.max, chars = True)
		#data = txt.getIOData(txtData, params)
		
		log("learn model")
		rnnlm.learn(lm, params, txtData)
		
		log("save model")
		lm.save(modelOutFile)
		
		log("done")
	else:
		raise Exception("AAAAA")
