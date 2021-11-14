import re, sys, os
from collections import Counter
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
import checkFasta
import numpy as np
import pandas as pd


USAGE = """
USAGE:
	python EGAAC.py input.fasta <sliding_window> <output>

	input.fasta:      the input protein sequence file in fasta format.
	sliding_window:   the sliding window, integer, defaule: 5
	output:           the encoding file, default: 'encodings.tsv'
"""

def EGAAC(fastas, window=5, **kw):
	if checkFasta.checkFasta(fastas) == False:
		print('Error: for "EGAAC" encoding, the input fasta sequences should be with equal length. \n\n')
		return 0

	if window < 1:
		print('Error: the sliding window should be greater than zero' + '\n\n')
		return 0

	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()

	encodings = []
	header = ['#']
	for w in range(1, len(fastas[0][1]) - window + 2):
		for g in groupKey:
			header.append('SW.'+str(w)+'.'+ g)

	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for j in range(len(sequence)):
			if j + window <= len(sequence):
				count = Counter(sequence[j:j + window])
				myDict = {}
				for key in groupKey:
					for aa in group[key]:
						myDict[key] = myDict.get(key, 0) + count[aa]
				for key in groupKey:
					code.append(myDict[key] / window)
		encodings.append(code)
	return encodings

kw = {'path': r"F:\lxj\研究\code\mycode-PPTPP\data\trainsets\SBP_1.txt",'order': 'ACDEFGHIKLMNPQRSTVWY'}

fastas = readFasta.readFasta(r"F:\lxj\研究\code\mycode-PPTPP\data\trainsets\SBP_1.txt")

sw=5
result = EGAAC(fastas, sw, **kw)



data1=np.matrix(result[1:])[:,1:]
data_=pd.DataFrame(data=data1)
data_.to_csv('EGAAC_SBP_train.csv', header=None, index=None)
