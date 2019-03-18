import pickle
import random
import math
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize
from preprocess import Preprocessor
from ngramlm import NGramLM

class Discriminator:
	
	def __init__(self, model_paths=None, models=None, smoothing=True, smoothing_k=1,\
				 unk_tackle=False, unk_threshold=1):
		self.model_paths = model_paths
		if model_paths != None:
			mods = []
			for path in model_paths:
				mods.append(self.open_model(path))
			self.models = mods
		else:
			self.models = models
		self.smoothing = smoothing
		self.smoothing_k = smoothing_k
		self.unk_tackle = unk_tackle
		self.unk_threshold = unk_threshold
		if self.smoothing == False:
			self.smoothing_k = 0
	
	def open_model(self, path):
		if path == None:
			raise RuntimeError("Invalid Model Path")
		with open(path, "rb") as f:
			model = pickle.load(f)
		return model
	
	def get_n_minus_one_gram_counts(self, model):
		grams = list()
		cc = model.cc
		for sentence in cc:
			for i in range(len(sentence) - model.n + 1):
				grams.append(" ".join(sentence[i : i - model.n + 1]))
		gc = Counter(grams)
		return gc
	
	def predict(self, sentence):
		# Works only for more-than-1-gram
		results = dict()
		models = self.models
		
		for i in range(len(models)):
			old_model_words = None
			m = models[i]
			if m.n == 1:
				raise RuntimeError("Unigram given as input")
			if self.unk_tackle == True and m.unk_tackle == False:
				# print("hi")
				if m.n == 2:
					words = [i for j in range(len(m.cc)) for i in m.cc[j]]
					words = Counter(words)
					old_model_words = words
					helper = lambda n: "UNK" if words[n.lower()] < self.unk_threshold else n.lower()
					new_model = dict()
					for word in words:
						if words[word] < self.unk_threshold:
							if "UNK" not in new_model and word in m.model:
								new_model['UNK'] = list()
							if word in m.model:
								clean_vals = [helper(tok) for tok in m.model[word] if helper(tok).isalpha()]
								new_model["UNK"].extend(clean_vals)
						else:
							if word not in new_model and word in m.model:
								new_model[word] = list()
							if word in m.model:
								clean_vals = [helper(tok) for tok in m.model[word] if helper(tok).isalpha()]
								new_model[word].extend(clean_vals)
					m.model = new_model
					
				else:

					m = NGramLM(corpus=m.corpus, n=m.n,\
										unk_tackle=True, unk_threshold=self.unk_threshold)
					m.fit()
				m.save_model("models/ut_model_"+str(i)+".pkl")
#             print(m == models[i])
			pp = Preprocessor()
			sentence = pp.remove_urls(sentence)
			cleaned_sent = pp.remove_punctuation(sentence)
			cleaned_words = pp.tokenize_clean(cleaned_sent)
			if self.unk_tackle == True:
				for i in range(len(cleaned_words)):
					if cleaned_words[i] not in m.model and cleaned_words[i] != "</s>":
						cleaned_words[i] = "UNK"
			# print(cleaned_words)
			gc = self.get_n_minus_one_gram_counts(m)
			gc_tot = len(list(self.get_n_minus_one_gram_counts(m).keys()))
			unk_count = 0
			n = m.n
			model = m.model
			prob = 0
			for i in range(len(cleaned_words) - n + 1):
				word = " ".join(cleaned_words[i : i + n - 1])
				multiplier = 0
				if word in model: 
					multiplier = (model[word].count(cleaned_words[i + n - 1]) + \
					   self.smoothing_k)/(len(model[word]) + (gc_tot * self.smoothing_k))
				else:
					if self.smoothing_k != 0:
						multiplier = (self.smoothing_k) / (gc_tot * self.smoothing_k)
					else:
						raise RuntimeError("Word out of voacabulary.")
				prob+=math.log(multiplier)
			results[m.corpus.split("/")[-2]] = [math.exp(prob), prob]
		keys = list(results.keys())
		keys.sort()
		# print(results)
		results = pd.DataFrame({"Class" : keys, "Class Probabilities" : [results[key][0] for key in keys], \
								"Log Class Probabilities" : [results[key][1] for key in keys]})
		results.set_index("Class", inplace=True)
		return results

