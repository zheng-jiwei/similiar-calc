import numpy as np 
import pandas as pd 
import spacy, sys, getopt, os, datetime

from gensim import models
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

nlp=spacy.load('ja_ginza')
for p in nlp.pipeline:
    print(p)

SET_ALPHA=0.0015
MODEL_DOC2VEC="./doc2vec.model"

def docs_to_sentences(texts, indexs):
    for index in range(len(texts)):
       result=nlp(str(texts.iloc[index]))
       noun_list = list(map(str, result.noun_chunks))
       yield TaggedDocument(words=noun_list, tags=[indexs[index]])


# type=1 -> all
# type=2 -> noun
# type=3 -> noun + adj + adv
def init_data(type=1):
    if os.path.isfile(MODEL_DOC2VEC):
        print("loading....%s" % datetime.datetime.now())
        model = Doc2Vec.load(MODEL_DOC2VEC)
        print("finish.... %s " % datetime.datetime.now())
    else:
        print("read data.......%s" % datetime.datetime.now())
        orgin_ds=pd.read_csv("../../../site-diff/output/site-content.csv", low_memory=False, index_col='site_host')
        obj_cols = [col for col in orgin_ds.columns if orgin_ds[col].dtype == "object"]
        ds = orgin_ds[obj_cols].copy().fillna("")
        ds = ds[:100]
        texts= ds['body'].str.cat([ds['meta_name_keywords'], ds['meta_name_description'], ds['body_keyword']], sep=' ')
        indexs = list(ds.index)
        print("read data.[end] %s" % datetime.datetime.now())
        print("training data.......%s" % datetime.datetime.now())
        docs = docs_to_sentences(texts, indexs)
        model=train_doc2vec_model(docs)
        print("training data [end] %s" % datetime.datetime.now())
        print("save to disk........%s" % datetime.datetime.now())
        model.save(MODEL_DOC2VEC)
        print("save to disk..[end] %s" % datetime.datetime.now())
    return model


def train_doc2vec_model(sentences):
#voctor_size: ベクトル化した際の次元数
#alpha: 学習率  -> 低いほど精度が高いですが、収束が遅くなります。
#sample: 単語を無視する際の頻度の閾値 -> あまりに高い頻度で出現する単語は意味のない単語である可能性が高いので、無視することがあります
#min_count: 学習に使う単語の最低出現回数
#workers: 学習時のスレッド数

    model = models.Doc2Vec(voctor_size=400, alpha=SET_ALPHA, sample=0.00001, min_count=1, workers=15, dm=0)
    model.build_vocab(sentences)
    for x in range(30):
        print("訓練 %d 回目" % x)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model


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
    output_count = "10"
    result = []

    try:
        opts, args = getopt.getopt(argv,"h:H:t:T:o",["host-name1=","host-name2=", "text_1=", "text_2=", "out_count="])
    except e:
        print(e, 'python similiar_spacy.py --out_count=<"all"|"-3"|"3" > -h <host-name> [-H <host_name>] | -t <text> [-T <text>]')
        sys.exit(2)
    for opt, arg in opts:
      if opt in ("-h", "--host-name1"):
         host_1 = arg
      elif opt in ("-H", "--host-name2"):
         host_2 = arg
      elif opt in ("-t", "--text_1"):
         text_1 = arg
      elif opt in ("-T", "--text_2"):
         text_2 = arg
      elif opt in ("-o", "--out_count"):
         output_count = arg

    model = init_data()
    if not text_1 == -1:
       if not text_2 == -1:
          words1, words2 = get_all_noun_from_doc(text_1, text_2)
          result = pd.DataFrame({"result": model.n_similarity(words1, words2)}, index=["none"])
       else:
          words1, words2 = get_all_noun_from_doc(text_1)
          word_vec = model.infer_vector(doc_words=list(map(str, words1)), steps=500, alpha=SET_ALPHA)
          result = model.docvecs.most_similar([word_vec], topn=int(output_count))
          result = pd.DataFrame({"result":[item[1] for item in result]}, index=[item[0] for item in result])
    elif not host_1 == -1:
       if not host_2 == -1:
          result = pd.DataFrame({"result":model.docvecs.similarity(host_1, host_2)}, index=[host_1+"<->"+host_2])
       else:
          result = model.docvecs.most_similar(host_1, topn=int(output_count))
          result = pd.DataFrame({"result":[item[1] for item in result]}, index=[item[0] for item in result])
    else:
       #result_df.to_csv("./result/spacy_result.csv")
       print("未実装")

    if len(result) > 0:
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


if __name__ == "__main__":
   main(sys.argv[1:])


