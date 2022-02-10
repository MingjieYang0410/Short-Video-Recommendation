import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings,os,gc
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec

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

data["userid"]="u"+data["userid"].astype(str)
data["feedid"]="i"+data["feedid"].astype(str)

G = nx.from_pandas_edgelist(data, "userid", "feedid", edge_attr=True, create_using=nx.Graph())


def get_randomwalk(node, path_length, G):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

all_nodes = list(G.nodes())


random_walks = []
for n in tqdm(all_nodes):
    for i in range(10):
        random_walks.append(get_randomwalk(n, 10, G))

model = Word2Vec(size=100, window=5, negative=10, alpha=0.03, min_alpha=0.0007,
                 workers=12, seed=9482, sg=1)
model.build_vocab(random_walks, progress_per=2)

model.train(random_walks, total_examples = model.corpus_count, epochs=5, report_delay=1)

temp = data.drop_duplicates('feedid')[["feedid"]].reset_index(drop=True)
for i in range(0, 100):
    col="feedid_{}_deepwalk".format(i)
    temp[col]=temp['feedid'].apply(lambda x:model.wv[x].tolist()[i])
temp['feedid'] = temp['feedid'].apply(lambda x:int(x[1:]))
temp.to_pickle(features_path+"{mc}_{gc}_{dim}.pickle".format(mc='feedid',gc="deepwalk",dim=str(100)))


temp = data.drop_duplicates('userid')[["userid"]].reset_index(drop=True)
for i in range(0, 100):
    col="userid{}_deepwalk".format(i)
    temp[col]=temp['userid'].apply(lambda x:model.wv[x].tolist()[i])
temp['userid'] = temp['userid'].apply(lambda x:int(x[1:]))
temp.to_pickle(features_path+"{mc}_{gc}_{dim}.pickle".format(mc='userid',gc="deepwalk",dim=str(100)))

data["authorid"]="a"+data["authorid"].astype(str)

G1 = nx.from_pandas_edgelist(data, "userid", "authorid", edge_attr=True, create_using=nx.Graph())

all_nodes = list(G1.nodes())

random_walks1 = []
for n in tqdm(all_nodes):
    for i in range(10):
        random_walks1.append(get_randomwalk(n,10, G1))

model = Word2Vec(size=100, window=5, negative=10, alpha=0.03, min_alpha=0.0007,
                 workers=12, seed=9482, sg=1)
model.build_vocab(random_walks1, progress_per=2)
model.train(random_walks1, total_examples = model.corpus_count, epochs=5, report_delay=1)

temp = data.drop_duplicates('authorid')[["authorid"]].reset_index(drop=True)
for i in range(0, 100):
    col="authorid_{}_deepwalk".format(i)
    temp[col]=temp['authorid'].apply(lambda x:model.wv[x].tolist()[i])
temp['authorid'] = temp['authorid'].apply(lambda x:int(x[1:]))
temp.to_pickle(features_path+"{mc}_{gc}_{dim}.pickle".format(mc='authorid',gc="deepwalk",dim=str(100)))


temp = data.drop_duplicates('userid')[["userid"]].reset_index(drop=True)
for i in range(0, 100):
    col="userid{}_deepwalk2".format(i)
    temp[col]=temp['userid'].apply(lambda x:model.wv[x].tolist()[i])
temp['userid'] = temp['userid'].apply(lambda x:int(x[1:]))
temp.to_pickle(features_path+"{mc}_{gc}_{dim}.pickle".format(mc='userid',gc="deepwalk2",dim=str(100)))

