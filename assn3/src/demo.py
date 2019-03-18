from discriminator import Discriminator
from generator import Generator
from ngramlm import NGramLM
# import pandas as pd
from preprocess import Preprocessor
import pickle

def main():
	# f = open("model/comp.graphics_uni.pkl")
	# Generating a unigram sentence
	# with open("models/comp.graphics_uni.pkl", "rb") as f:
	# 	model = pickle.load(f)
	# print(len(model.model.keys()))
	graph_uni_gen = Generator(model_path="models/comp.graphics_uni.pkl")
	bikes_uni_gen = Generator(model_path="models/rec.motorcycles_uni.pkl")
	g_unigram_sentence = graph_uni_gen.generate_sentence(threshold=50)
	b_unigram_sentence = bikes_uni_gen.generate_sentence(threshold=50)

	# Generating a bigram sentence
	graph_bi_gen = Generator(model_path="models/comp.graphics_bi.pkl")
	bikes_bi_gen = Generator(model_path="models/rec.motorcycles_bi.pkl")
	g_bigram_sentence_1 = graph_bi_gen.generate_sentence(threshold=50)
	b_bigram_sentence_1 = bikes_bi_gen.generate_sentence(threshold=50)
	g_bigram_sentence_2 = graph_bi_gen.generate_sentence(threshold=50)
	b_bigram_sentence_2 = bikes_bi_gen.generate_sentence(threshold=50)

	graph_tri_gen = Generator(model_path="models/comp.graphics_tri.pkl")
	bikes_tri_gen = Generator(model_path="models/rec.motorcycles_tri.pkl")
	g_trigram_sentence_1 = graph_tri_gen.generate_sentence(threshold=50)
	b_trigram_sentence_1 = bikes_tri_gen.generate_sentence(threshold=50)
	g_trigram_sentence_2 = graph_tri_gen.generate_sentence(threshold=50)
	b_trigram_sentence_2 = bikes_tri_gen.generate_sentence(threshold=50)	


	print("Corpus\t:\t" + graph_uni_gen.model.corpus)
	print("\nGenerated sentences\n")
	print("Unigram -- \n")
	print(g_unigram_sentence + "\n")
	print("Bigram -- \n")
	print("1. " + g_bigram_sentence_1 + "\n")
	print("2. " + g_bigram_sentence_2 + "\n")
	print("Trigram -- \n")
	print("1. " + g_trigram_sentence_1 + "\n")
	print("2. " + g_trigram_sentence_2 + "\n")


	print("===============================================================================")
	print("===============================================================================\n")

	print("Corpus\t:\t" + bikes_uni_gen.model.corpus)
	print("\nGenerated sentences\n")
	print("Unigram -- \n")
	print(b_unigram_sentence + "\n")
	print("Bigram -- \n")
	print("1. " + b_bigram_sentence_1 + "\n")
	print("2. " + b_bigram_sentence_2 + "\n")
	print("Trigram -- \n")
	print("1. " + b_trigram_sentence_1 + "\n")
	print("2. " + b_trigram_sentence_2 + "\n")


	print("===============================================================================")
	print("===============================================================================\n")

	print("\n\n")
	print("Without UNK replacement")

	d1 = Discriminator(model_paths=["models/comp.graphics_bi.pkl", "models/rec.motorcycles_bi.pkl"]\
		, smoothing=True, smoothing_k=1, unk_tackle=False)

	print("Predicting sentence:\n", g_bigram_sentence_1,"\n")
	print("Results\n")
	print(d1.predict(" ".join(g_bigram_sentence_1.split()[1:-1])))

	print("\n\n")

	print("Predicting sentence:\n", b_bigram_sentence_2,"\n")
	print("Results\n")
	print(d1.predict(" ".join(b_bigram_sentence_2.split()[1:-1])))

	print("\n\n")

	sent = input("Enter a sentence:\n").strip()
	print("Predicting sentence:\n", sent,"\n")
	print("Results\n")
	print(d1.predict(sent))
	print("\n\n")
	print("With UNK Replacement")
	print("\n\n")
	d2 = Discriminator(model_paths=["models/comp.graphics_bi.pkl", "models/rec.motorcycles_bi.pkl"]\
		, smoothing=True, smoothing_k=1, unk_tackle=True, unk_threshold=10)
	print("Predicting sentence:\n", g_bigram_sentence_1,"\n")
	print("Results\n")
	print(d2.predict(" ".join(g_bigram_sentence_1.split()[1:-1])))

	print("\n\n")

	print("Predicting sentence:\n", b_bigram_sentence_2,"\n")
	print("Results\n")
	print(d2.predict(" ".join(b_bigram_sentence_2.split()[1:-1])))

	print("\n\n")
	sent = input("Enter a sentence:\n").strip()
	print("Predicting sentence:\n", sent,"\n")
	print("Results\n")
	print(d2.predict(sent))

	print("\n\n")

if __name__ == '__main__':
	main()









