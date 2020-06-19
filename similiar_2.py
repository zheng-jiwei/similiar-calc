import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyknp import Juman
import sys
from os import listdir, path
from gensim import models
from gensim.models.doc2vec import TaggedDocument

ds = pd.read_csv("./csv-data/site-content.csv", index_col="site_host")


if path.isfile("./doc2vec.model"):
    model = models.Doc2Vec.load('doc2vec.model')
    size = len(ds.index)-1
    print(size)
    data = model.docvecs.most_similar('www.connecty.co.jp', topn=size)
    print(data)
    exit()



# intとobjectの clolumnsを取得する
int_cols = [col for col in ds.columns if ds[col].dtype in ["int64", "float64"]]
obj_cols = [col for col in ds.columns if ds[col].dtype == "object"]

# int columnの nilは 0で差し替える
new_int_ds = ds[int_cols].copy().fillna(0)

#nil のobject columnをexpendする
append_ds = pd.DataFrame()
for col in obj_cols:
    append_ds[col + "_is_none"] = ds[col].isnull()
new_all_ds = pd.concat([new_int_ds, append_ds], axis="columns")

# object columnの nilは ""で差し替える
new_obj_ds = ds[obj_cols].copy().fillna("")
new_all_ds = pd.concat([new_all_ds, new_obj_ds], axis="columns")


#!wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/jumanpp/jumanpp-1.01.tar.xz
#!tar xvf jumanpp-1.01.tar.xz
#!cd jumanpp-1.01
#!./configure
#!make
#!make install

#wget http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/knp/knp-4.20.tar.bz2
#tar xvf knp-4.20.tar.bz2
#cd knp-4.20
#./configure
#make
#sudo make install

#!pip install pyknp


jumanapp = Juman()

def corpus_files():
    dirs = [path.join('./text', x)
            for x in listdir('./text') if not x.endswith('.txt')]
    docs = [path.join(x, y)
            for x in dirs for y in listdir(x) if not x.startswith('LICENSE')]
    return docs


def read_document(path):
    with open(path, 'r') as f:
        return f.read()

def subs_string(input, size):
    result=input.encode('utf-8')
    try:
        result = result[0:size].decode('utf-8')
    except Exception as e:
        result = -1
    return result


def split_into_words(text):
  try:
    text=text.replace(" ", "")
    text=text.replace("　", "")
    orignal_size = len(text.encode('utf-8'))
    target_size = 4096
    if len(text.encode('utf-8')) > target_size:
        for index in range(4):
            result = subs_string(text, target_size)
            target_size = target_size - 1
            if not result == -1:
               text = result
               break
    print(orignal_size, "->", len(text.encode('utf-8')))
    result = jumanapp.analysis(text)
    return [mrph.midasi for mrph in result.mrph_list()]
  except Exception as e:
    print(text)
    exit()

def doc_to_sentence(doc, name):
    words = split_into_words(doc)
    #return LabeledSentence(words=words, tags=[name])
    return TaggedDocument(words=words, tags=[name])
 

def corpus_to_sentences(docs, corpus):
#    docs   = [read_document(x) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        print('\r前処理中 {}/{}'.format(idx+1, len(corpus)))
        yield doc_to_sentence(doc, name)


#corpus = corpus_files()
#sentences = corpus_to_sentences(corpus)

target_data = ds["body"].to_numpy().tolist()

sentences = corpus_to_sentences(target_data, ds.index.to_numpy().tolist()[:len(target_data)])
model = models.Doc2Vec(list(sentences), dm=1, vector_size=300, window=15, alpha=.025, min_alpha=.025, min_count=1, sample=1e-6)

print('\n訓練開始')
for epoch in range(20):
    print('Epoch: {}'.format(epoch + 1))
    model.train(sentences, total_examples=len(ds.index), epochs=epoch)
    model.alpha -= (0.025 - 0.0001) / 20
    model.min_alpha = model.alpha


print("\n完成")

model.save('doc2vec.model')
#model = models.Doc2Vec.load('doc2vec.model')

#model.docvecs.similarity('www.connecty.co.jp', 'sports.dunlop.co.jp')
size = len(ds.index)-1
output = model.docvecs.most_similar('www.connecty.co.jp', topn=size)
print(output)

