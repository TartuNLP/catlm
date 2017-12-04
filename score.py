#!/usr/bin/env python3

import sys
import rnnlm
from datetime import datetime

if __name__ == "__main__":
	try:
		dataFile = sys.argv[1]
		dictInFile = sys.argv[2]
		modelInFile = sys.argv[3]
	except:
		print("Usage: score.py  data_file  dict_file  model_file")
	else:
		model = rnnlm.loadModels(modelInFile, dictInFile)
		
		#with open(dictInFile, 'rb') as fh:
		#	metadict = pickle.load(fh)                                                                      
		#
		#model = rnnlm.initModel(metadict['v'], metadict['m'], embSize = 16, hdnSize = 256)                 
		#model.load_weights(modelInFile + "w") 
		
		#mdl = (model, metadict)
		
		textData = rnnlm.file2text(dataFile, chars = True)
		
		start = datetime.now()
		
		for snt in textData:
			output = "".join(snt) + str(rnnlm.score(snt, model))
			print(output.encode(encoding = 'utf8'))
			#print(snt, rnnlm.score(snt, mdl1))
			
			newtime = datetime.now()
			
			print(str(newtime - start))
			
			start = newtime
