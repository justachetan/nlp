import os
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

class Preprocessor:
	
	'''Class to preprocess a corpus'''
	def __init__(self, corpus=None, replace=[]):
		self.corpus = corpus
		self.replace = replace
	
	def remove_header_meta_data(self, text):
		'''Assumes that metadata ends with two newlines'''
		text = text.split('\n\n')
		return "\n\n".join(text[1:])

	def remove_footer_meta_data(self, text):
		'''Assumes that footer ends with a --'''
		text = text.split("--")
		if len(text) > 1:
			text = text[:-1]
		text = "--".join(text)
		return text

	def remove_urls(self, text):
		'''Assumes domain names to be two - five characters'''
		return re.sub(r'(http)?\S*@?\S*\.\S{2,5}\s*', " ", text)

	def remove_punctuation(self, text):
		return re.sub(r'[!\"#$%&\'\(\)\*+,-\./:;<=>?@\[\\\]^_`\{|\}~]', "", text)

	def clean_tokens(self, tokens):
		helper = lambda n: "UNK" if n.lower() in self.replace else n.lower()
		cleaned_toks = [helper(tok) for tok in tokens if helper(tok).isalpha()]
		# for tok in tokens:
		# 	if tok.isalpha() and tok not in self.replace:
		# 		cleaned_toks.append(tok.lower())
		return cleaned_toks

	def tokenize_clean(self, sentence):
		tokens = word_tokenize(sentence)
		cleaned_tokens = self.clean_tokens(tokens)
		cleaned_tokens = ['<s>'] + cleaned_tokens + ['</s>']
		return cleaned_tokens

	def preprocess(self, data):
		data = self.remove_header_meta_data(data)
		data = self.remove_footer_meta_data(data)
		data = self.remove_urls(data)
		sentences = sent_tokenize(data)
		cleaned_sent = [self.remove_punctuation(s) for s in sentences]
		cleaned_words = [self.tokenize_clean(s) for s in cleaned_sent]
		return cleaned_words
	
	def preprocess_corpus(self):
		preprocessed_sentences = []
		files = os.listdir(self.corpus)
		for file in files:
			with open(self.corpus + file, "r", encoding='ISO-8859-1') as f:
				data = f.read()
				preprocessed_doc = self.preprocess(data)
				preprocessed_sentences.extend(preprocessed_doc)
		return preprocessed_sentences, len(files)
	
	
	
