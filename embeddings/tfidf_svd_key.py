import os,gc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

data_path = "../data/original/"
features_path = "../data/features/"

feed_info = pd.read_csv(data_path+'feed_info.csv')
user_action = pd.read_csv(data_path+'user_action.csv')
feed_emb = pd.read_csv(data_path+'feed_embeddings.csv')
test = pd.read_csv(data_path+'test_a.csv')

feed_info = feed_info[["feedid","authorid","manual_keyword_list","machine_keyword_list"]]
feed_info[["feedid","authorid"]] = feed_info[["feedid","authorid"]].fillna(-1)
feed_info['key1'] = feed_info['manual_keyword_list'].fillna('x').apply(lambda x: str(x).replace(";"," "))
feed_info['key2'] = feed_info['machine_keyword_list'].fillna('x').apply(lambda x: str(x).replace(";"," "))

data = user_action[["userid","feedid"]].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
temp = test[["userid","feedid"]]
data = pd.concat([data, temp], axis=0).reset_index(drop=True)

data = data.merge(feed_info[["feedid","authorid","key1","key2"]], on='feedid', how='left')

## 将一个用户观看过的所有视频的描述向量拼接起来全部一起去描述这个用户！manual_keyword_list
dim_n = 128
data["key1"] = data["key1"].astype(str)
user_key1 = data.groupby("userid")["key1"].agg(list).reset_index()
user_key1["key1"] = user_key1["key1"].apply(lambda x: " ".join(x))
user_key1.columns = ["userid","key1"]

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(user_key1["key1"])
tfidf_feat = tfidf_vectorizer.transform(user_key1["key1"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["user_key1"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([user_key1[["userid"]], df_svd], axis=1)
df_feat.to_pickle(features_path+"user_key1_vec_{dim}.pickle".format(dim=str(dim_n)))

data["key2"] = data["key2"].astype(str)
user_key2 = data.groupby("userid")["key2"].agg(list).reset_index()
user_key2["key2"] = user_key2["key2"].apply(lambda x: " ".join(x))
user_key2.columns = ["userid","key2"]

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(user_key2["key2"])
tfidf_feat = tfidf_vectorizer.transform(user_key2["key2"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["user_key2"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([user_key2[["userid"]], df_svd], axis=1)
df_feat.to_pickle(features_path+"user_key2_vec_{dim}.pickle".format(dim=str(dim_n)))

data["key1"] = data["key1"].astype(str)
tmp = data.groupby("authorid")["key1"].agg(list).reset_index()
tmp["key1"] = tmp["key1"].apply(lambda x: " ".join(x))
tmp.columns = ["authorid","key1_tmp"]

feed_info["key1"] = feed_info["key1"].astype(str)
author_key1 = feed_info.groupby("authorid")["key1"].agg(list).reset_index()
author_key1["key1"] = author_key1["key1"].apply(lambda x: " ".join(x))
author_key1.columns = ["authorid","key1"]
author_key1 = author_key1.merge(tmp,how='left',on="authorid").fillna('x')
author_key1["key1"] = author_key1["key1"] + ' ' +author_key1["key1_tmp"]

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(author_key1["key1"])
tfidf_feat = tfidf_vectorizer.transform(author_key1["key1"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["author_key1"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([author_key1[["authorid"]], df_svd], axis=1)
df_feat.to_pickle(features_path+"author_key1_vec_{dim}.pickle".format(dim=str(dim_n)))

data["key2"] = data["key2"].astype(str)
tmp = data.groupby("authorid")["key2"].agg(list).reset_index()
tmp["key2"] = tmp["key2"].apply(lambda x: " ".join(x))
tmp.columns = ["authorid","key2_tmp"]

feed_info["key2"] = feed_info["key2"].astype(str)
author_key2 = feed_info.groupby("authorid")["key2"].agg(list).reset_index()
author_key2["key2"] = author_key2["key2"].apply(lambda x: " ".join(x))
author_key2.columns = ["authorid","key2"]
author_key2 = author_key2.merge(tmp,how='left',on="authorid").fillna('x')
author_key2["key2"] = author_key2["key2"] + ' ' +author_key2["key2_tmp"]

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(author_key2["key2"])
tfidf_feat = tfidf_vectorizer.transform(author_key2["key2"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["author_key2"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([author_key2[["authorid"]], df_svd], axis=1)
df_feat.to_pickle(features_path+"author_key2_vec_{dim}.pickle".format(dim=str(dim_n)))

data["key1"] = data["key1"].astype(str)
tmp = data.groupby("feedid")["key1"].agg(list).reset_index()
tmp["key1"] = tmp["key1"].apply(lambda x: " ".join(x))
tmp.columns = ["feedid","key1_tmp"]

feed_info["key1"] = feed_info["key1"].astype(str)
feed_key1 = feed_info.groupby("feedid")["key1"].agg(list).reset_index()
feed_key1["key1"] = feed_key1["key1"].apply(lambda x: " ".join(x))
feed_key1.columns = ["feedid","key1"]
feed_key1 = feed_key1.merge(tmp,how='left',on="feedid").fillna('x')
feed_key1["key1"] = feed_key1["key1"] + ' ' +feed_key1["key1_tmp"]

print(tmp.shape)
print(feed_key1.shape)

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(feed_key1["key1"])
tfidf_feat = tfidf_vectorizer.transform(feed_key1["key1"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["feed_key1"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([feed_key1[["feedid"]], df_svd], axis=1)

df_feat.to_pickle(features_path+"feed_key1_vec_{dim}.pickle".format(dim=str(dim_n)))

data["key2"] = data["key2"].astype(str)
tmp = data.groupby("feedid")["key2"].agg(list).reset_index()
tmp["key2"] = tmp["key2"].apply(lambda x: " ".join(x))
tmp.columns = ["feedid","key2_tmp"]

feed_info["key2"] = feed_info["key2"].astype(str)
feed_key2 = feed_info.groupby("feedid")["key2"].agg(list).reset_index()
feed_key2["key2"] = feed_key2["key2"].apply(lambda x: " ".join(x))
feed_key2.columns = ["feedid","key2"]
feed_key2 = feed_key2.merge(tmp,how='left',on="feedid").fillna('x')
feed_key2["key2"] = feed_key2["key2"] + ' ' +feed_key2["key2_tmp"]

print(tmp.shape)
print(feed_key2.shape)

del tmp, feed_key2["key2_tmp"]
gc.collect()

print("vectorizer...")
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 1),min_df=3,max_df=0.94)
tfidf_vectorizer.fit(feed_key2["key2"])
tfidf_feat = tfidf_vectorizer.transform(feed_key2["key2"])

print("TruncatedSVD...")
svd = TruncatedSVD(n_components=dim_n, n_iter=5, random_state=2009)
svd.fit(tfidf_feat)
tfidf_svd = svd.transform(tfidf_feat)
df_svd = pd.DataFrame(tfidf_svd, columns=["feed_key2"+'_svd'+str(i) for i in range(dim_n)])
df_feat = pd.concat([feed_key2[["feedid"]], df_svd], axis=1)
df_feat.to_pickle(features_path+"feed_key2_vec_{dim}.pickle".format(dim=str(dim_n)))