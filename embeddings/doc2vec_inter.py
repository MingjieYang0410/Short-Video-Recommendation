import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings,os,gc

data_path = "../data/original/"
features_path = "../data/features/"

feed_info = pd.read_csv(data_path+'feed_info.csv')
user_action = pd.read_csv(data_path+'user_action.csv')
feed_emb = pd.read_csv(data_path+'feed_embeddings.csv')
test = pd.read_csv(data_path+'test_a.csv')

data = user_action[["userid","feedid"]].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
test = test[["userid","feedid"]]
data = pd.concat([data, test], axis=0).reset_index(drop=True)
data = data.merge(feed_info[["feedid","authorid"]], how='left',on="feedid")


def doc_to_vec(df, main_col, gp_col, dim_n):
    print(main_col, gp_col)
    df[gp_col] = df[gp_col].astype(str)
    tmp = df.groupby(main_col)[gp_col].agg(list).reset_index()
    df[gp_col] = df[gp_col].astype(int)
    tmp[gp_col] = tmp[gp_col].apply(lambda x: [str(i) for i in x])

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tmp[gp_col])]  # 给每个文档（用户交互序列）一个唯一的ID
    model = Doc2Vec(documents, workers=20, min_count=5, vector_size=dim_n, window=25, dm=0)
#    model.train(documents, total_examples=model.corpus_count, epochs=10)

    vectors = model.docvecs.vectors_docs
    df_w2v = pd.DataFrame(vectors, columns=['{mc}_{gc}_d2v'.format(mc=main_col[:-2], gc=gp_col[:-2]) + str(i) for i in
                                            range(dim_n)])
    df_feat = pd.concat([tmp[[main_col]], df_w2v], axis=1)
    df_feat.to_pickle(
        features_path + "{mc}_{gc}_d2v_{dim}.pickle".format(mc=main_col[:-2], gc=gp_col[:-2], dim=str(dim_n)))

    del tmp, df_w2v, df_feat, documents, vectors
    gc.collect()


dim_n = 150
for main_col, gp_col in [('userid', 'feedid'), ('userid', 'authorid'), ('feedid', 'userid'), ('authorid', 'userid')]:
    doc_to_vec(data,main_col,gp_col,dim_n)
