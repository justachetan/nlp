import random
import pickle
from collections import Counter
from preprocess import Preprocessor

class NGramLM:
	
	'''The class to train the n-gram generator model'''
	
	def __init__(self, corpus=None, n=1, unk_tackle=False, unk_threshold=0):
		self.n = n
		self.model = dict()
		self.corpus = corpus
		self.unk_tackle = unk_tackle
		self.unk_threshold = unk_threshold
		self.cc = None
	
	def train_unigram_model(self, cc=None):
		model = dict()
		words = list()
		if cc is None:
			raise RuntimeError('Corpus not specified')
		for sent in cc:
			for word in sent:
				if word != '<s>':
					words.append(word)
		model['<s>'] = words
		return model
			
	def train_ngram_model(self, n=2, cc=None):
		model = dict()
		if cc is None:
			raise RuntimeError('Corpus not specified')
		for i in range(len(cc)):
			sent = cc[i]
			for j in range(len(sent) - n + 1):
				word = " ".join(sent[j:j + n - 1])
				if word not in model:
					model[word] = list()
#                 print(j, n, j+n, len(sent))
				model[word].append(sent[j + n - 1])
		return model
	
	def fit(self):
		pp = Preprocessor(self.corpus)
		cc, num = pp.preprocess_corpus()

		# print(len(cc))

		if self.unk_tackle == True:
			# print("hi")
			words = [i for j in range(len(cc)) for i in cc[j]]
			words = Counter(words)
			remove_words = [i for i in list(words.keys()) if words.get(i) < self.unk_threshold]
			pp = Preprocessor(self.corpus, remove_words)
			cc, num = pp.preprocess_corpus()
		# print("UNK" in [i for j in range(len(cc)) for i in cc[j]])

		self.cc = cc
		if self.n == 1:
			self.model = self.train_unigram_model(cc)
		else:
			self.model = self.train_ngram_model(self.n, cc)
	
	def save_model(self, out="model.pkl"):
		with open(out, "wb") as output:
			pickle.dump(self, output)

def main():

	graph_uni = NGramLM(corpus="data/20_newsgroups/comp.graphics/", n=1)
	graph_uni.fit()
	graph_uni.save_model("models/comp.graphics_uni.pkl")
	graph_bi = NGramLM(corpus="data/20_newsgroups/comp.graphics/", n=2)
	graph_bi.fit()
	graph_bi.save_model("models/comp.graphics_bi.pkl")
	graph_tri = NGramLM(corpus="data/20_newsgroups/comp.graphics/", n=3)
	graph_tri.fit()
	graph_tri.save_model("models/comp.graphics_tri.pkl")


	bikes_uni = NGramLM(corpus="data/20_newsgroups/rec.motorcycles/", n=1)
	bikes_uni.fit()
	bikes_uni.save_model("models/rec.motorcycles_uni.pkl")
	bikes_bi = NGramLM(corpus="data/20_newsgroups/rec.motorcycles/", n=2)
	bikes_bi.fit()
	bikes_bi.save_model("models/rec.motorcycles_bi.pkl")
	bikes_tri = NGramLM(corpus="data/20_newsgroups/rec.motorcycles/", n=3)
	bikes_tri.fit()
	bikes_tri.save_model("models/rec.motorcycles_tri.pkl")

if __name__ == '__main__':
	main()