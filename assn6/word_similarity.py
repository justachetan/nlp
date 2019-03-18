from gensim.models import KeyedVectors


file = "../models/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(file, binary=True)

result = model.most_similar(positive=["Delhi", "China"], negative=["India"], topn=10)


print("1.",result)

result2 = model.most_similar(positive=["ISRO", "USA"], negative=["India"], topn=10)

print("2.", result2)







