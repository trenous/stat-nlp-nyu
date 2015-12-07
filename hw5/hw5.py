import numpy as np
import random
import numpy.random as nrand
import pickle
import argparse
import operator
import math
import scipy.spatial.distance as distance
import time
import scipy.stats as st
import numpy.linalg as lina
import os.path as path
import gensim
starting_alpha = 0.025
unigram_table = []
data = []
dict_str_int = {} 
dict_int_str = {}
dict_word_vec = {}
W = []
V = []
vocab_size = 0
layer_1 = 100
numTokens = 0
unigramCounts = {}
STOP = '<S>'
sample = 1e-5
MIN_FREQ = 15
min_alpha = 1e-4
win = 4
k = 3
max_iter = 1
table_size = 100000000

#Reads Data as Numpy Array
def readDatawv(fileName, maxSentences):
	global data, numSen
	data = []
	file = open(fileName, 'r')
	i = 0
	numSen = 0
	while (numSen < maxSentences+1):
		line = file.readline()
		if line == "":
			break
		sen  = []
		for word in line.split():
			i += 1
			word = word.lower()
			sen.append(word)
		data.append(sen)
		numSen += 1
	print("Sentences read: " + str(numSen))
	print("Tokens read: " + str(i))


def wv(maxSentences):
	f = open('1', 'r')
	iterator = myclass('1', maxSentences)
	x = time.time()
	model = gensim.models.Word2Vec(iterator, size=layer_1, window=win,negative=0, min_count=MIN_FREQ, sample=sample, workers=4, hs=1, iter=max_iter, min_alpha=min_alpha)
	print(time.time() - x)
	testwv(model)
	return model

class myclass(object):
	def __init__(self, f, maxsen):
		self.f =f
		self.maxsen = maxsen
	def __iter__(self):
		inp = open(self.f, 'r')
		maximumSen = self.maxsen
		while (True):
			data = []
			numSen = 0
			if maximumSen == 0:
				break
			while (numSen < min(2500000, maximumSen)):
				line = inp.readline()
				if not line:
					return
				data.append(line.split())
				numSen += 1
			maximumSen = maximumSen - numSen
			for sentence in data:
				yield sentence


def testwv(mod):
	fw = open('./combined.csv', 'r')
	line = fw.readline()
	print("First Line: " + line)
	wordsimpairs = []
	i = 0
	for line in fw:
		if i == 0:
			print("First Line WordsimSet: " + line)
		i+=1
		words = line.split(',')
		wordsimpairs.append([words[0].lower(),words[1].lower(),float(words[2])])	
	corr = evaluatewv(wordsimpairs, mod)
	print("Evaluation on wordsim353 dataset: Evaluating on " + str(len(wordsimpairs)) + " Wordpairs.")
	print("Spearman Rank Correlation: " + str(corr))

def evaluatewv(wordSims, mod):
	sims_gold = []
	sims_embed = []
	not_found = 0

	for tup in wordSims:
		if (mod.__contains__(tup[0]) and mod.__contains__(tup[1])):
			sims_gold.append(tup[2])
			sims_embed.append(mod.similarity(tup[0],tup[1]))
		else:
			not_found += 1
	print(str(not_found) + " Word pairs were not found")
	print("Evaluating on " + str(len(sims_gold)) + " Word Pairs")
	return st.spearmanr(sims_gold, sims_embed)

#Reads Data as Numpy Array
def readData(fileName, maxSentences):
	global data, numTokens
	data = []
	data.append(STOP)
	file = open(fileName, 'r')
	i = 1
	sen = 0
	while (sen < maxSentences):
		line = file.readline()
		if not line:
			break
		for word in line.split():
			data.append(word)
			i += 1
		data.append(STOP)
		sen += 1
		i += 1
	numTokens = i
	print("Tokens read: " + str(i))


def cleanData():
	global data, unigramCounts, numTokens
	unigramCounts = {}
	for i in range (0, numTokens):
		if data[i] in unigramCounts:
			unigramCounts[data[i]] += 1.0
		else:
			unigramCounts[data[i]] = 1.0

	j = 0
	clean_data = []
	for i in range (0, numTokens):
		if data[i] not in unigramCounts:
			continue
		if (unigramCounts[data[i]] >= MIN_FREQ):
			if (unigramCounts[data[i]] > sample*numTokens and data[i] != STOP):
				rand = random.random()
				if (rand < 1 - math.sqrt((sample*numTokens)/unigramCounts[data[i]])):
					continue
			clean_data.append(data[i])
			j += 1
		else:
			del unigramCounts[data[i]]
	del unigramCounts[STOP]
	data = clean_data
	numTokens = j

#Creates the Dictionary for the Data
def makeDict():
	global dict_int_str, dict_str_int, vocab_size, unigramCounts
	dict_str_int = {}
	dict_int_str = {}
	dict_str_int[STOP] = 0
	dict_int_str[0] =  STOP
	i = 1
	for word in data:
		if word not in dict_str_int:
			dict_str_int[word] = i
			dict_int_str[i] = word
			i += 1
	vocab_size = i
	unigramCounts = dict((dict_str_int[key] , unigramCounts[key]/ float(numTokens)) for i,key in enumerate(unigramCounts))
	


# transforms the data to Integer array using the dictionary
def toIntArray():
	global data
	data_int = []
	for i in range (0,numTokens):
		data_int.append(dict_str_int[data[i]])
	data = data_int

# creates a table for sampling of Unigrams
def uni():
	global table_size
	global unigram_table
	unigram_table = []
	unigramProbs = dict((key , pow(unigramCounts[key], 0.75)) for i,key in enumerate(unigramCounts))
	sorted_counts =  sorted(unigramProbs.items(), key=operator.itemgetter(1))
	cell_weight = sorted_counts[0][1]
	print("Cell Weght: " + str(cell_weight))
	word_num = 0.0
	word_index = 0
	word = sorted_counts[word_index]
	print(word)
	word_total = unigramProbs[word[0]]
	for i in range (0,table_size):
		unigram_table.append(word[0])
		word_num += 1.0
		if (cell_weight * word_num) >= word_total:
			word_index += 1
			if (word_index >= len(sorted_counts)):
				table_size = i+1
				print("TableSize " + str(table_size))
				print("TableSize " + str(len(unigram_table)))
				break
			word = sorted_counts[word_index]
			word_total = unigramProbs[word[0]]
			word_num = 0.0
	print(table_size)
	


# Initializes a matrix of dimension dxp
def initializeUniformMatrix():
	global W,V
	W = []
	V = []
	for i in range (0,vocab_size):
		W.append(nrand.uniform(-1.0/(2*layer_1),1.0/(2*layer_1),(layer_1)))
		V.append(np.zeros((layer_1)))


# Computes the minimal cosine similarity between two rows of the matrices W0 and W1
def getMaxCosineDistance(W0, W1):
	dist = 0.0
	dist_sum = 0.0
	for i in range (1,vocab_size):
		cos = distance.cosine(W0[i], W1[i])
		dist_sum += cos
		if (cos > dist) :
			dist = cos
	print("Avg Cos Distance: " + str(dist_sum/vocab_size))
	return dist


def getCosineSimilarity(w1, w2):
	v = dict_word_vec[w1]
	w = dict_word_vec[w2]
	return v.dot(w) / (lina.norm(v)*lina.norm(w))

def getCosineSimilarity_vec(v, w):
	return v.dot(w) / (lina.norm(v)*lina.norm(w))

def closest_word_vec(vec):
	sim = -1
	closest = vec
	for k,v in dict_word_vec.iteritems():
		cos = getCosineSimilarity_vec(v,vec)
		if cos == 1:
			continue
		if (cos > sim):
			sim = cos
			closest = k
	return sim, closest

def ten_closest(word):
	return get_ten_closest(dict_word_vec[word])

def get_ten_closest(vec):
	cl  = ""
	for i in range(0,10):
		sim = -1
		closest = ""
		for k,v in dict_word_vec.iteritems():
			if np.array_equal(v,vec):
				continue
			if k in cl:
				continue
			cos = getCosineSimilarity_vec(v,vec)
			if (cos > sim):
				closest = k
				sim = cos
		cl += "<" + str(closest) + " : " + str(sim) + "> "
	return cl

def ten_farthest(word):
	return get_ten_farthest(dict_word_vec[word])

def get_ten_farthest(vec):
	cl = {}
	for i in range(0,10):
		sim = 1
		farthest = 0
		for k,v in dict_word_vec.iteritems():
			if k in cl:
				continue
			cos = getCosineSimilarity_vec(v,vec)
			if (cos < sim):
				farthest = k
				sim = cos
		cl[farthest] = sim
	return cl


def closest_word(word):
	sim = -1
	closest = word
	for k,v in dict_word_vec.iteritems():
		if k == word:
			continue
		cos = getCosineSimilarity(k,word)
		if (cos > sim):
			sim = cos
			closest = k
	return sim, closest

def farthest_word(word):
	sim = 1
	farthest = word
	for k,v in dict_word_vec.iteritems():
		if k == STOP:
			continue
		cos = getCosineSimilarity(k,word)
		if (cos < sim):
			sim = cos
			farthest = k
	return sim, farthest

def vector(w):
	return dict_word_vec[w]

def evaluate(wordSims):
	global dict_word_vec
	sims_gold = []
	sims_embed = []
	not_found = 0

	for tup in wordSims:
		if (tup[0] in dict_word_vec and tup[1] in dict_word_vec):
			sims_gold.append(tup[2])
			sims_embed.append(getCosineSimilarity(tup[0],tup[1]))
		else:
			print("Not Found: " + tup[0] + ' , ' + tup[1])
			not_found += 1
	print(str(not_found) + " Word pairs were not found")
	return st.spearmanr(sims_gold, sims_embed)

def test():
	fw = open('./combined.csv', 'r')
	line = fw.readline()
	print("First Line: " + line)
	wordsimpairs = []
	i = 0
	for line in fw:
		if i == 0:
			print("First Line WordsimSet: " + line)
		i+=1
		words = line.split(',')
		wordsimpairs.append([words[0].lower(),words[1].lower(),float(words[2])])	
	corr = evaluate(wordsimpairs)
	print("Evaluation on wordsim353 dataset: Evaluating on " + str(len(wordsimpairs)) + " Wordpairs.")
	print("Spearman Rank Correlation: " + str(corr))

def makewordvec():
	global dict_word_vec
	dict_word_vec = {}
	for i in range (0,vocab_size):
		word = dict_int_str[i]
		dict_word_vec[word] = W[i]

def cbow(num):
	global W,V, data, dict_word_vec, min_alpha, numTokens
	#parse arguments
	maxSents = num
	f = '1'
	# Read Data into Array
	readData(f,maxSents)
	# discard infrequent words ; subsample frequent words
	cleanData()
	# Create two-way Dictionary
	makeDict()
	# Create Integer representation of Data
	toIntArray()
	# Create Unigram Distribution of Data raised to the power of 0.75
	uni()
	# Initialize W and W'
	initializeUniformMatrix()
	# Stochastic Gradient Descent
	epoch_length = 10000
	iter = 0
	epoch = 0
	random.seed()
	alpha = starting_alpha
	W1 = np.copy(W)
	V1 = np.copy(V)
	word_count = 0
	print("Starting SGA, Num Tokens:" + str(numTokens) + ", vocab size: " + str(vocab_size))
	while iter < max_iter:
		wc_iter = 0
		x = time.time()
		while wc_iter < numTokens-1:
			this_epoch = min(wc_iter + epoch_length, numTokens-1)
			for index in range(wc_iter, this_epoch):
				word = data[index]
				if (word == 0):
					continue
				# Stochastic window size
				s = int(round(random.random() * win))
				if s == 0:
					continue
				s += 1
				cw = 0
				# Collect the words in the sentence that this context is associated with
				context = np.zeros(layer_1)
				for j in range (1,s):
					c = data[index - j]
					if (c == 0):
						break	
					cw += 1	
					context += W[c]
				for j in range (1,s):
					c = data[index + j]
					if (c == 0):
						break
					cw += 1
					context += W[c]
				if cw == 0:
					continue
				context /= cw
				# Compute approximate gradients for each word
				wprime = np.zeros(layer_1)
				dot = context.dot(V1[word])
				if dot <= 20:
					incorpus_weight = alpha / (1+math.exp(dot))
					wprime += V1[word] * incorpus_weight
					V1[word] += context * incorpus_weight
				# negative Sampling
				for j in range (0,k):
					rand = random.random()
					rand *= table_size
					negword = unigram_table[int(rand)]
					if word == negword:
						continue
					dot = W1[negword].dot(context)
					if dot < -20:
						continue
					excorpus_weight = alpha / (1+math.exp(-dot))
					wprime -= V1[negword] * excorpus_weight
					V1[negword] -= context * excorpus_weight
				for j in range (1,s):
					c = data[index - j]
					if (c == 0):
						break	
					W[c] += wprime
			 	for j in range (1,s):
					c = data[index + j]
					if (c == 0):
						break
					W1[c] += wprime
			epoch += 1
			word_count += epoch_length
			wc_iter = this_epoch
			alpha = starting_alpha * (1 - float(word_count)/(max_iter * numTokens) )
			if alpha < min_alpha:
				alpha = min_alpha			
			if (epoch % 10 == 0):
				print("Time Taken: " + str(time.time() - x))
				print ("epoch: " + str(epoch) + "/" + str(numTokens/epoch_length))
				print("alpha: " + str(alpha))
		iter += 1
		epoch = 0
		# Test for convergence
		dist = max(getMaxCosineDistance(W1,W),getMaxCosineDistance(V1,V))
		print ("cosine Distance: " + str(dist))
		W = np.copy(W1)
		V = np.copy(V1)
		if (dist < 10e-3):
			print("converged")
			break
		print("iter: " + str(iter))


def read():
	global dict_word_vec
	f = open('7-12958550-6', 'r')
	dict_word_vec = pickle.load(f)

def main(num):
	global W,V, data, dict_word_vec, min_alpha, numTokens
	#parse arguments
	maxSents = num
	f = '1'
	# Read Data into Array
	readData(f,maxSents)
	# discard infrequent words ; subsample frequent words
	cleanData()
	# Create two-way Dictionary
	makeDict()
	# Create Integer representation of Data
	toIntArray()
	# Create Unigram Distribution of Data raised to the power of 0.75
	uni()
	# Initialize W and W'
	initializeUniformMatrix()
	# Stochastic Gradient Descent
	epoch_length = 10000
	iter = 0
	epoch = 0
	random.seed()
	alpha = starting_alpha
	W1 = np.copy(W)
	V1 = np.copy(V)
	word_count = 0
	print("Starting SGA, Num Tokens:" + str(numTokens) + ", vocab size: " + str(vocab_size))
	while iter < max_iter:
		wc_iter = 0
		x = time.time()
		while wc_iter < numTokens-1:
			this_epoch = min(wc_iter + epoch_length, numTokens-1)
			for index in range(wc_iter, this_epoch):
				context = data[index]
				if (context == 0):
					continue
				# Stochastic window size
				s = int(round(random.random() * win))
				if s == 0:
					continue
				s += 1
				# Collect the words in the sentence that this context is associated with
				words = []
				for j in range (1,s):
					word = data[index - j]
					if (word == 0):
						break		
					words.append(word)
				for j in range (1,s):
					word = data[index + j]
					if (word == 0):
						break
					words.append(word)
				# Compute approximate gradients for each word
				for word in words:
					wprime = np.zeros(layer_1)
					dot = V1[context].dot(W1[word])
					if dot <= 20:
						incorpus_weight = alpha / (1+math.exp(dot))
						wprime += V1[context] * incorpus_weight
						V1[context] += W1[word] * incorpus_weight
					# negative Sampling
					for j in range (0,k):
						rand = random.random()
						rand *= table_size
						negcontext = unigram_table[int(rand)]
						if context == negcontext:
							continue
						dot = V1[negcontext].dot(W1[word])
						if dot < -20:
							continue
						excorpus_weight = alpha / (1+math.exp(-dot))
						wprime -= V1[negcontext] * excorpus_weight
						V1[negcontext] -= W1[word] * excorpus_weight
					W1[word] += wprime
			epoch += 1
			word_count += epoch_length
			wc_iter = this_epoch
			alpha = starting_alpha * (1 - float(word_count)/(max_iter * numTokens) )
			if alpha < min_alpha:
				alpha = min_alpha			
			if (epoch % 10 == 0):
				print("Time Taken: " + str(time.time() - x))
				print ("epoch: " + str(epoch) + "/" + str(numTokens/epoch_length))
				print("alpha: " + str(alpha))
		iter += 1
		epoch = 0
		# Test for convergence
		dist = max(getMaxCosineDistance(W1,W),getMaxCosineDistance(V1,V))
		print ("cosine Distance: " + str(dist))
		W = np.copy(W1)
		V = np.copy(V1)
		if (dist < 10e-3):
			print("converged")
			break
		print("iter: " + str(iter))

		


	# Create Dictionary mapping Strings to vectors			
	data = 0
	dict_word_vec = {}
	for i in range (1,vocab_size):
		word = dict_int_str[i]
		dict_word_vec[word] = W[i]
	# Test
	test()
	return dict_word_vec

if __name__ == "__main__":
    main()
