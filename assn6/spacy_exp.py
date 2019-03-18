#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy


# In[2]:


nlp = spacy.load('en_core_web_sm')


# In[3]:


doc = "test.txt"


# In[4]:


fh = open(doc)


# In[5]:


data = fh.read().strip()


# In[6]:


toks = nlp(data)
print("--------------------------------------------------------------------------------------------------------\n\n")

print("DATA GIVEN")
print("--------------------------------------------------------------------------------------------------------\n\n")
print(data)
print("--------------------------------------------------------------------------------------------------------\n\n")


# In[7]:


# POS TAGGING
print("POS TAGGING\n")
for t in toks:
    print(t, "\t\t", t.pos_)
print("--------------------------------------------------------------------------------------------------------\n\n")


# In[8]:


# LEMMATIZATION
print("LEMMATIZATION\n")
for t in toks:
    print(t, "\t", t.lemma_)
print("--------------------------------------------------------------------------------------------------------\n\n")


# In[9]:


# NER
print("NER\n")
for t in toks.ents:
    print(t.text, "\t\t", t.label_)
print("--------------------------------------------------------------------------------------------------------\n\n")


# In[10]:


word_doc = "word_test.txt"


# In[11]:


fw = open(word_doc)
print("WORD SIMILARITY EXPERIMENT\n")


# In[12]:


for line in fw:
    toks = nlp(line.strip())
    print("Word 1:", toks[0], "\t", "Word 2:", toks[2], "\tSimilarity: ", toks[0].similarity(toks[1]))
print("--------------------------------------------------------------------------------------------------------\n\n")


# In[ ]:





# In[ ]:




