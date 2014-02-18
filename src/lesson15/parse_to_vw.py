import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import itertools

def get_all_text(j):
  j = json.loads(j)
  if j:
    title = j.get('title', "")
    body = j.get('body', "")
    if title and body:
      return title.encode('ascii', 'ignore') + body.encode('ascii', 'ignore')
    elif body:
      return body
    else:
      return ""
  else:
    return ""

def save_vocab(vocab, vocab_file):
  f = open(vocab_file, 'w')
  for (word, id) in vocab.items():
    f.write(word.encode('ascii', 'ignore') + '\t' + str(id) + "\n")
  f.close()

def main(input_file = 'train.tsv'):
  data = pd.read_csv(input_file, sep='\t', error_bad_lines=False)
  docs = data.boilerplate.map(lambda x: get_all_text(x))
  vec = CountVectorizer(stop_words='english', lowercase=True)

  vw_file = open(input_file+".vw", 'w')
  encoded_docs = vec.fit_transform(docs).tocoo()
  vw_docs = defaultdict(str)
  for i,j,v in itertools.izip(encoded_docs.row, encoded_docs.col, encoded_docs.data):
    vw_docs[i] += " " + str(j) + ":" + str(v)

  vocab = dict( (word, id) for (id, word) in enumerate(vec.get_feature_names()))
  for doc in vw_docs.values():
    vw_file.write("|" + doc)
    vw_file.write("\n")
 
  save_vocab(vocab, input_file+'.vocab')

if __name__ == '__main__':
  main()
