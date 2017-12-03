import numpy as np

from collections import defaultdict
from para import Params, Data

SOS = 1
EOS = 2
OOV = 3

def loadFile(filename, maxLen = 50, chars = False):
	if filename == '-':
		fh = sys.stdin
	else:
		fh = open(filename, 'r')
	
	result = []
	
	tokFreqs = defaultdict(int)
	catFreqs = defaultdict(lambda: defaultdict(int))
	
	for line in fh:
		fields = line.strip().split("\t")
		text = fields[-1]
		cats = fields[:-1]
		
		if chars:
			toks = list(text)
		else:
			toks = text.split()
		
		#update freq dict
		for tok in toks:
			tokFreqs[tok] += 1
		
		#update cat dicts
		for i, cat in enumerate(cats):
			catFreqs[i][cat] += 1
		
		if toks and len(toks) < maxLen:
			result.append({ 'text': toks, 'cats': cats })
	
	if filename != '-':
		fh.close()
	
	return result, tokFreqs, catFreqs

def catfreqs2dicts(catFreqs):
	return [{ k: idx for (idx, k) in enumerate(catFreqs[i]) } for i in range(len(catFreqs))]

def freqs2dicts(tokFreqs, vocSize = None):
	idx2word = { 0: None, SOS: "__s__", EOS: "__/s__", OOV: "UNK" }
	word2idx = dict(zip(idx2word.values(), idx2word.keys()))
	
	if vocSize is None:
		vocSize = len(tokFreqs) + 4 # None, SOS, EOS, OOV
	
	endIdx = vocSize - len(idx2word)
	
	#sorting and truncating
	for tok in sorted(tokFreqs, key=lambda x: -tokFreqs[x])[:endIdx]:
		idx = len(idx2word)
		
		word2idx[tok] = idx
		idx2word[idx] = tok
	
	return word2idx, idx2word
	
def getIOData(textData, para): #word2idx, cats2idx, maxLen):
	numSnts = len(textData)
	
	vocSize = len(para.w2i)
	
	txtInputs = np.zeros([numSnts, para.max, vocSize], dtype='int32')
	catInputs = [np.zeros([numSnts, para.max, len(para.c2i[i])], dtype='int32') for i in range(len(para.c2i))]
	
	outputs = np.zeros([numSnts, para.max, 1], dtype='int32')
	
	for i, line in enumerate(textData):
		txtInputs[i, 0, SOS] = 1
		
		for k, cat in enumerate(line['cats']):
			catIdx = para.c2i[k][cat]
			
			catInputs[k][i, 0, catIdx] = 1
		
		for j, tok in enumerate(line['text']):
			try:
				idx = para.w2i[tok]
			except KeyError:
				idx = OOV
			
			txtInputs[i, j+1, idx] = 1
			outputs[i, j, 0] = idx
			
			for k, cat in enumerate(line['cats']):
				catIdx = para.c2i[k][cat]
				
				catInputs[k][i, j+1, catIdx] = 1
		
		outputs[i, len(line['text']), 0] = EOS
	
	return Data(txtInputs, catInputs, outputs)

def loadAndClean(filename, maxLen, chars = False, vocSize = None):
	txtData, tokFreqs, catFreqs = loadFile(filename, maxLen, chars = chars)
	
	w2i, i2w = freqs2dicts(tokFreqs, vocSize = vocSize)
	c2i = catfreqs2dicts(catFreqs)
	
	params = Params(maxLen, w2i, i2w, c2i)
	
	data = getIOData(txtData, params)
	
	return data, params
