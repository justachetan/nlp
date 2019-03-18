from collections import Counter
import pickle
import numpy as np
import argparse

# python hmm.py ./test.txt -m HMM.pkl


def get_hmm(corpus=None, f=None, save=False, load=False, unk=False, unk_lim=10):
	
	HMM = dict()

	if load == True:
		with open(f, "rb") as fh:
			HMM = pickle.load(fh)
		return HMM
	
	if corpus == None:
		raise RuntimeError("Please enter a valid filename.")
	
	file = open(corpus, "r")
	data = file.read().strip()
	sentences = data.split("\n\n")
	
	
	A = dict()
	B = dict()
	pi = list()

	for x in range(len(sentences)):
		
		s = sentences[x]
		obs = [toks.split("\t")[0] for toks in s.split("\n")]
		tags = [toks.split("\t")[1] for toks in s.split("\n")]
		
		pi.append(tags[0])
		
		for i in range(0, len(tags) - 1):
			if tags[i] not in A:
				A[tags[i]] = list()
			A[tags[i]].append(tags[i+1])
			
			if tags[i] not in B:
				B[tags[i]] = list()
			B[tags[i]].append(obs[i])
	
	for key in A:
		A[key] = Counter(A[key])
		total = sum(A[key].values())

	for key in B:
		B[key] = Counter(B[key])
		if unk == True:
			B[key]["UNK"] = 0
			to_pop = list()
			for s in B[key].keys():
				if B[key][s] < unk_lim:
					B[key]["UNK"] += B[key][s]
					to_pop.append(s)
			for i in to_pop: B[key].pop(i)

	
	
	pi = Counter(pi)
	total_pi = sum(pi.values())
	for p in pi:
		pi[p] = pi[p]/total_pi
	
	HMM["A"] = A
	HMM["B"] = B
	HMM["pi"] = pi
	
	if save == True:
		pickle.dump(HMM, open("HMM.pkl", "wb"), protocol=2)
	
	return HMM
	 




def decode(inp, HMM, smooth=True, k=1):
	# takes input as a list of tokens from sentence
	
	A = HMM["A"]
	B = HMM["B"]
	pi = HMM["pi"]
	
	T = len(inp)
	Q = len(HMM["A"].keys())
	
	trellis = np.zeros((Q, T))
	bt = np.zeros((Q, T))
	
	mapping = dict()
	tags = list(A.keys())
	for i in range(Q):
		mapping[i] = tags[i]
	
	for i in range(Q):
		trellis[i, 0] = pi[mapping[i]] * ((B[mapping[i]][inp[0]] + k) /                                           (sum(B[mapping[i]].values()) + len(tags) * k))
		bt[i, 0] = 0
		
	for j in range(1, T):
		
		for i in range(Q):
			
			trellis[i, j] = np.max([trellis[l, j-1] *                                     ((A[mapping[l]][mapping[i]] + k) / (sum(A[mapping[l]].values()) + k * Q)) *                                     ((B[mapping[i]][inp[j]] + k) / (sum(B[mapping[i]].values()) + Q * k))                                     if inp[j] in B[mapping[i]]                                    else                                     trellis[l, j-1] *                                     ((A[mapping[l]][mapping[i]] + k) / (sum(A[mapping[l]].values()) + k * Q)) *                                     ((B[mapping[i]]['UNK'] + k) / (sum(B[mapping[i]].values()) + Q * k))                                     for l in range(Q)])
			bt[i, j] = np.argmax([trellis[l, j-1] *                                   ((A[mapping[l]][mapping[i]] + k) / (sum(A[mapping[l]].values()) + k * Q)) *                                   ((B[mapping[i]][inp[j]] + k) / (sum(B[mapping[i]].values()) + Q * k))                                   if inp[j] in B[mapping[i]]                                  else                                  trellis[l, j-1] *                                   ((A[mapping[l]][mapping[i]] + k) / (sum(A[mapping[l]].values()) + k * Q)) *                                   ((B[mapping[i]]['UNK'] + k) / (sum(B[mapping[i]].values()) + Q * k))                                   for l in range(Q)])
	
	
	
	path = list()
	maxi = np.argmax([trellis[l, T - 1] * ((A[mapping[l]]['.'] + k)/(sum(A[mapping[l]].values()) + k * Q)) for l in range(Q)])
	path.append(maxi)
	i = int(maxi)
	for j in range(T - 1, 0, -1):
		path = [int(bt[i, j])] + path
		i = int(bt[i, j])
	return [mapping[s] for s in path] + ["."]




def training_acc(train_doc=None, HMM=None): 
	
	if train_doc==None:
		raise RuntimeError("Please enter a valid input document.")
	
	if HMM==None:
		raise RuntimeError("Please provide an HMM object.")
		
	fh = open(train_doc, "r")
	data = fh.read()
	
	sentences = data.split("\n\n")
	
	correct = 0
	total = len(sentences)
	
	for s in sentences:
		
		tags = [toks.split("\t")[1] for toks in s.split("\n")]
		words = [toks.split("\t")[0] for toks in s.split("\n")][:-1]
		print(s)
		predicted = decode(words, HMM)
		
		if decode == predicted: correct+=1
	
	return correct / total





def predict(test_doc=None, HMM=None, out=None):
	
	if test_doc == None:
		raise RuntimeError("Provide a valid test document.")
	
	if HMM == None:
		raise RuntimeError("Please provide an HMM object.")
	fo = None
	if out != None:
		fo = open(out, "a+")
	
	fh = open(test_doc)
	
	data = fh.read().strip()
	
	sentences = data.split("\n\n")
	
	for s in sentences:
		
		words = s.split("\n")
		tags = decode(words[:-1], HMM)
		string = ""
		for i in range(len(words)):
			string = string + words[i] + "\t" + tags[i] + "\n"
		
		
		if out == None:
			print(string)
		else:
			print(string, file=fo)

def main():

	parser = argparse.ArgumentParser(description="This is a simple HMM that \
		trains on an input document and can then decode a hidden sequence using \
		the Viterbi algorithm.")

	parser.add_argument("input", help="Path to test file in the prescribed format.", type=str)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-m", "--model", help="Path to pickled HMM.", default=None)
	group.add_argument("-ts", "--train", help="Path to Training corpus", default=None)
	parser.add_argument("-u", "--unk", help="Threshold for replacing emissions with 'UNK'. Default is 1", default=1, type=int)
	parser.add_argument("-k", help="k value for add-k smoothening. Default is 1", default=1, type=int)
	parser.add_argument("-s", "--save", help="Path where to save trained model. If set to Y then a file, 'HMM.pkl' is created in the current directory.", default=0, choices=[0, 1], type=int)
	parser.add_argument("-o", "--out", help="Where to output the result. Provide a filepath. By default, the output is displayed at stdout.", default=None, type=str)

	args = parser.parse_args()

	HMM = None

	if args.input == None:

		parser.error("Must provide the test file as input.")

	if args.train == None:

		HMM = get_hmm(f=args.model, load=True)

	if args.model == None:

		save = False
		if args.save == 1: save = True
		HMM = get_hmm(corpus=args.train, save=save, unk=True, unk_lim=args.unk)



	predict(HMM=HMM, out=args.out, test_doc=args.input)


if __name__ == '__main__':
	main()











