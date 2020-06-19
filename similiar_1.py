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

#int_cols.extend(obj_cols)
#そのたtypeの columnsを取得する
#left_ds = ds.copy().drop(columns=int_cols)

from sklearn.preprocessing import LabelEncoder
lable_enc = LabelEncoder()
ds_1 = new_all_ds.copy()

for col in obj_cols:
    ds_1[col] = lable_enc.fit_transform(ds_1[col])

data_array = ds_1.to_numpy().tolist()

#全ての hostnameを取る
indexs = ds_1.index
def get_similar_map(fun_call):
    result_df = pd.DataFrame()
    for index in range(len(data_array)):
        result = list()
        compare_x = data_array[index]
        for line in data_array:
            result.append(fun_call(line, compare_x))
        result_df[indexs[index]] = result
    result_df.index = ds_1.index
    return result_df


def pearsonSimilar(inA,inB):  
    if len(inA)<3:  
        return 1.0  
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]  


def euclidSimilar(inA,inB):  
    return 1.0/(1.0+np.linalg.norm(np.array(inA)-np.array(inB)))


def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)


result_df = get_similar_map(pearsonSimilar)
result_df.to_csv("./result/site_pearson_result.csv")

result_df = get_similar_map(euclidSimilar)
result_df.to_csv("./result/site_euclid_result.csv")

result_df = get_similar_map(cosSimilar)
result_df.to_csv("./result/site_cos_result.csv")

import matplotlib.pyplot as plt
import seaborn as sns

graf = plt.figure(figsize=(40,40))

# Add title
plt.title("similiar of site")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=result_df, annot=True)

# Add label for horizontal axis
plt.xlabel("site_host")

graf.savefig("test.png")
