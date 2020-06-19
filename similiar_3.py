import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


ds = pd.read_csv("./csv-data/site-content.csv", index_col="site_host")

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


from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load("../jawiki.doc2vec.dbow300d.model")

from janome.tokenizer import Tokenizer
def sep_by_janome(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    return docs

documents = new_all_ds["body"].to_numpy().tolist()
document_vecs=[]
for d in documents:
    document_vecs.append(model.infer_vector(sep_by_janome(d)))


target_text = ds.loc["www.connecty.co.jp"]["body"]
tokens = sep_by_janome(target_text)
input_vec = model.infer_vector(tokens)

#コサイン類似度の計算＋ランキング化
import numpy as np
rank_size = 5
 
v1 = np.linalg.norm(input_vec)
cos_sim = []
for v2 in document_vecs:
    cos_sim.append( np.dot(input_vec,v2)/(v1*np.linalg.norm(v2)) )
doc_sort = np.argsort(np.array(cos_sim))[::-1]
cos_sort = sorted(cos_sim,reverse=True)
 
for i in range(rank_size):
    print(cos_sort[i])
    print(documents[doc_sort[i]])
