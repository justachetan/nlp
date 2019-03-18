#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nltk
import random
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


# In[2]:


DIR_LINK = "../data/20_newsgroups/"
tags = os.listdir(DIR_LINK)

tagged_corpora = list()
doc_list = list()

for i in range(len(tags)):
    docs = os.listdir(DIR_LINK + "/" + tags[i] + "/")
    for doc in docs:
        fh = open(DIR_LINK + "/" + tags[i] + "/" + doc, "rb")
        data = fh.read().strip()
        doc_list.append(simple_preprocess(data))
        tdoc = TaggedDocument(simple_preprocess(data), [tags[i]])
        fh.close()
        tagged_corpora.append(tdoc)


# In[3]:


model = Doc2Vec(vector_size=100, min_count=2, epochs=100)


# In[4]:


model.build_vocab(tagged_corpora)


# In[5]:


reqi = tags.index("comp.graphics")


# In[6]:


graphics_indices = [doc_list[tagged_corpora.index(i)] for i in tagged_corpora if i.tags[0] == tags[reqi]]
graphics_docs = random.sample(graphics_indices, 20)


# In[7]:


tags_found = []
diverse_docs = []
for i in range(len(tagged_corpora)):
    if tagged_corpora[i].tags[0] not in tags_found and tagged_corpora[i].tags[0] != tags[reqi]:
        tags_found.append(tagged_corpora[i].tags[0])
        diverse_docs.append(doc_list[i])


# In[8]:


src = model.infer_vector(graphics_docs[0])
gvecs = [model.infer_vector(i) for i in graphics_docs[1:]]
gcoss = model.wv.cosine_similarities(src, gvecs)


# In[9]:


dvecs = [model.infer_vector(i) for i in diverse_docs]
dcoss = model.wv.cosine_similarities(src, dvecs)


# In[10]:


print("Mean cosine similarity of graphics docs:", gcoss.mean())
print("Mean cosine similarity of diverse docs:", dcoss.mean())


# In[11]:


print("Are graphics documents more similar to a graphics doc on an avg.?:", gcoss.mean() > dcoss.mean())

