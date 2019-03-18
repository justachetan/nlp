# Aditya Chetan
# 2016217

import re
import string
import sys


# Regex global constants 

PARAGRAPH_REGEX = "(\n|\r|\t| {4,}){2,}\t*"
SENTENCE_REGEX = "([a-zA-Z\.]{3,}?|[0-9]+)[\.\'\"\?!\})\]]+\s+[\"(\[\{A-Z0-9\n]"
WORD_REGEX = "[^\[\{\}\.\"\)\(\]\!\?\b\s,]*[A-Z0-9a-z]+,?[^\[\}\{\'\"\)\(\]\!\?\b\s,]*"
WORD = sys.argv[1]
FILE = sys.argv[2]

# Reading the file

fh = open(FILE, "r")
data = fh.read()

# Searching paragraphs

para_matches = re.finditer(PARAGRAPH_REGEX, data)
para_matched_strings = [data[match.start(): match.end()] for match in para_matches]


# Searching sentences

sen_matches = re.finditer(SENTENCE_REGEX, data)
sen_matched_strings = [data[match.start(): match.end()] for match in sen_matches]

# Searching words

raw_words = re.finditer(WORD_REGEX, data)
raw_words_matched = [data[match.start(): match.end()] for match in raw_words]



# Counting word occurences

ovall = 0
startswith = 0
endswith = 0

start_regex = "^(" + WORD.capitalize() + "|" + WORD.upper()+ ")|[\.\?\!\"\'\\b\s]+\s\\b" + WORD.capitalize() + "\\b" + "|[\.\?\!\"\'\\b\s]*" + WORD.upper() + "\\b\s"
end_regex = "\\b" + WORD.lower() + "[\.\?\!\]\)\"\']+?" + "|\\b" + WORD + "[\.\?\!\]\)\"\']+?" + "|\\b" + WORD.upper() + "[\.\?\!\]\)\"\']+?"
ovall_regex = "\\b" + WORD.lower() + "\\b|\\b" + WORD.capitalize() + "\\b|\\b" + WORD.upper() + "\\b"
print(start_regex)
startswith = len(re.findall(start_regex, data))
endswith = len(re.findall(end_regex, data))
ovall = len(re.findall(ovall_regex, data))

print("Number of paragraphs:", len(para_matched_strings) )
print("Number of sentences:", len(sen_matched_strings))
print("Number of words:", len(raw_words_matched))
print("Number of senteces starting with " + WORD + ":", startswith)
print("Number of senteces ending with " + WORD + ":", endswith)
print("Count of " + WORD + ":", ovall)
