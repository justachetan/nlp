import pickle
import random
from ngramlm import NGramLM


class Generator:
	
	def __init__(self, model_path=None, model=None):
		self.model_path = model_path
		self.model = model
		if self.model == None:
			self.open_model()
		# print(self.model)

	
	def open_model(self):
		if self.model_path == None:
			raise RuntimeError("Invalid Model Path")
		with open(self.model_path, "rb") as f:
			self.model = pickle.load(f)
		return 
	
	def generate_sentence(self, threshold=50):
		sent = ""
		n = self.model.n
		model = self.model.model
		if n == 1:
			sent = sent + "<s>"
			for i in range(1, 50):
				new_word = random.choice(model["<s>"])
				sent = sent + " " + new_word
				if new_word == "</s>":
					break
		
		elif n != 1:
			count = 0
			start_choices = list()
			for word in model.keys():
				if "<s>" in word:
					count+=1
			to_start = random.random()
			for word in model.keys():
				if "<s>" in word:
					start_choices.append((len(model[word])/count, word))
			start_choices.sort()
			for i in range(0, len(start_choices)):
				if start_choices[i][0] > to_start:
					sent = sent + start_choices[i][1]
					# print(sent, "before start 1")
					break

			if sent == '':
				# print(sent, "before start 2")
				sent = sent + random.choice(start_choices)[1]
			
			
			for i in range(n - 1, 50):
				# print(sent)
				# print(" ".join(sent.split()[-n + 1:]))
				new_word = random.choice(model[" ".join(sent.split()[-n + 1:])])
				sent = sent + " " + new_word
				if new_word == "</s>":
					break
		if sent.split()[-1] != "</s>":
			sent = sent + " " + '</s>'
		return sent

