import numpy as np 
import pandas as pd 
import spacy, pickle, sys, getopt, os, datetime
from gensim.models import KeyedVectors

from spacy.tokens import Doc
from spacy.vocab import Vocab

nlp=spacy.load('ja_ginza')
for p in nlp.pipeline:
    print(p)


DIC_FILE = "./ginza_docs.dic"
DIC_INDEX="./ginaz_docs.index"


'''
wv = KeyedVectors.load_word2vec_format('./cc.ja.300.vec.gz', binary=False)
nlp.vocab.reset_vectors(width=wv.vectors.shape[1])
for word in wv.vocab.keys():
    nlp.vocab[word]
    nlp.vocab.set_vector(word, wv[word])
'''


def calculate_similarity(docs, doc, indexs):
    result = []
    for index, obj in enumerate(docs):
       if len(obj) == 0 or len(doc) == 0:
            result.append(0)
       else:
           result.append(doc.similarity(obj))
    return pd.DataFrame({'result': result}, index=indexs)


# type=1 -> all
# type=2 -> noun
# type=3 -> noun + adj + adv
def init_data(type=1):
    if os.path.isfile(DIC_FILE):
        print("loading....%s" % datetime.datetime.now())
        docs, indexs = load_docs_from_disk()
        print("finish.... %s " % datetime.datetime.now())
        return docs, indexs
    else:
        print("creating....%s" % datetime.datetime.now())
        orgin_ds=pd.read_csv("../../../site-diff/output/site-content.csv", low_memory=False, index_col='site_host')
        obj_cols = [col for col in orgin_ds.columns if orgin_ds[col].dtype == "object"]
        ds = orgin_ds[obj_cols].copy().fillna("")
        ds = ds[:100]
        texts= ds['body']
        print(ds.info())
        indexs = list(ds.index)
        docs=list(nlp.pipe(texts))
        print("finish......%s" % datetime.datetime.now())
        print("save to disk...%s" % datetime.datetime.now())
        save_docs_to_disk(docs, indexs)
        print("finish.........%s" % datetime.datetime.now())
    return docs, indexs

def save_docs_to_disk(docs, indexs):
    doc_bytes = [doc.to_bytes(exclude=['tensor', 'user_data']) for doc in docs]
    vocab_bytes = nlp.vocab.to_bytes()
    with open(DIC_FILE,"wb") as f:
        pickle.dump( (doc_bytes, vocab_bytes), f)
    with open(DIC_INDEX,"w") as f:
        f.write(",".join(indexs))
    return True

def load_docs_from_disk():
    with open(DIC_FILE, "rb") as f:
        doc_bytes, vocab_bytes = pickle.load(f)
    nlp.vocab.from_bytes(vocab_bytes)
    docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
    with open(DIC_INDEX, "r") as f:
        indexs = f.read().split(",")
    return docs, indexs


def get_all_noun_from_doc(text1, text2=""):
    doc1 = nlp(text1)
    noun1 = doc1.noun_chunks
    if not text2 == "":
       doc2 = nlp(text2)
       noun2 = doc2.noun_chunks
    else:
       noun2 = []
    return noun1, noun2

def main(argv):
    host_1 = -1
    host_2 = -1
    text_1 = -1
    text_2 = -1
    type = "all"
    output_count = "10"
    result = []
    result_df = pd.DataFrame()

    try:
        opts, args = getopt.getopt(argv,"h:H:t:T:o",["host-name1=","host-name2=", "text1=", "text2=", "out_count="])
    except e:
        print(e, 'python similiar_spacy.py --out_count=<"all"|"-3"|"3" > -h <host-name> [-H <outputfile>] | -t <text1> [-T <text2>]')
        sys.exit(2)

    for opt, arg in opts:
      if opt in ("-h", "--host-name1"):
         host_1 = arg
      elif opt in ("-H", "--host-name2"):
         host_2 = arg
      elif opt in ("-t", "--text1"):
         text_1 = arg
      elif opt in ("-T", "--text2"):
         text_2 = arg
      elif opt in ("-o", "--out_count"):
         output_count = arg
      elif opt in ("-p", "--type"):
         type = arg
    
    if not text_1 == -1:
       if not text_2 == -1:
          result = pd.DataFrame({"result":nlp(text_1).similarity(nlp(text_2))}, index=["none"])
       else:
          all_docs, indexs = init_data();
          in_doc = nlp(text_1)
          result = calculate_similarity(all_docs, in_doc, indexs)
    elif not host_1 == -1:
       if not host_2 == -1:
          all_docs, indexs = init_data();
          id_1 = indexs.index(host_1)
          id_2 = indexs.index(host_2)
          result = pd.DataFrame({"result": all_docs[id_1].similarity(all_docs[id_2])}, index=[host_1 + "<->" + host_2])
       else:
          all_docs, indexs = init_data();
          id_1 = indexs.index(host_1)
          in_doc = all_docs[id_1]
          result = calculate_similarity(all_docs, in_doc, indexs)
    else:
       all_docs, indexs = init_data();
       for index, compare_x in enumerate(all_docs):
          result = list()
          column = indexs[index]
          print("比較site %s" % column)
          for doc in all_docs:
             if len(doc) == 0 or len(compare_x) == 0:
                 result.append(0)
             else:
                 result.append(doc.similarity(compare_x))
          result_df[column] = result
       result_df.index = indexs
       result_df.to_csv("./result/spacy_result.csv")
       print("./result/spacy_result.csvを確認してください")

    if (len(result_df) > 0):
        print(result_df.head())
    elif len(result) > 0:
       if output_count == "all" or not text_2 == -1 or not host_2 == -1:
           print(result.sort_values(by="result", ascending=False))
       elif int(output_count) > 0:
           print("先頭の %d 件" % int(output_count))
           print(result.sort_values(by="result", ascending=False).head(int(output_count)))
       else:
           print("末尾の %d 件" % abs(int(output_count)))
           print(result.sort_values(by="result", ascending=True).head(abs(int(output_count))))
    else:
       print("no result")

def test_1():
   result=nlp.pipe([["セールス領域とテック領域におけるアウトソーシングや人材派遣サービスを提供し","aaa"],["放課後等デイサービス、就労支援、就労移行支援、就労継続支援Ｂ型等の施設の立ち上げ", "bb"]], as_tuples=True)
   docs = []
   for data in result:
      list_data = list(data)
      title = list_data[1]
      doc = list_data[0]
      doc.cats = title
      docs.append(doc)
   save_docs_to_disk(docs, ["aa", "bb"])

def test_2():
   docs, indexs = load_docs_from_disk()
   for doc in docs:
      print(doc.cats)

if __name__ == "__main__":
   main(sys.argv[1:])


