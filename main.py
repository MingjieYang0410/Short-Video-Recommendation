from sklearn.model_selection import KFold
from utils import *
from models import *
import warnings
import os
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


setup_seed(16)

data_path = "./data/original/"
features_path = "./data/features/"
models_path = "./data/models/"
label_cols = ['read_comment','like','click_avatar','forward','favorite','comment','follow']+['play']
res_path = "./data/results/"

feed_info = pd.read_csv(data_path+'feed_info.csv')
user_action = pd.read_csv(data_path+'user_action.csv')
feed_emb = pd.read_csv(data_path+'feed_embeddings.csv')


# 建立 id 2 index 映射字典
def id_encode(series):
    unique = list(series.unique())
    unique.sort()
    return dict(zip(unique, range(series.nunique()))) #, dict(zip(range(series.nunique()), unique))


if not os.path.exists(features_path+"userid2index.npy"):
    userid2index = id_encode(user_action.userid) # 所有 user_id 都有唯一对应的 index
    feedid2index = id_encode(feed_info.feedid) # 所有 feed_id 都有唯一对应的 index
    authorid2index = id_encode(feed_info.authorid)
    np.save(features_path+"userid2index.npy",userid2index)
    np.save(features_path+"feedid2index.npy",feedid2index)
    np.save(features_path+"authorid2index.npy",authorid2index)
else:
    print("yes")
    userid2index = np.load(features_path+"userid2index.npy",allow_pickle=True).item()
    feedid2index = np.load(features_path+"feedid2index.npy",allow_pickle=True).item()
    authorid2index = np.load(features_path+"authorid2index.npy",allow_pickle=True).item()


def read_id_dict(name, dim_n, emb_mode):
    tmp = pd.read_pickle(features_path + '{name}_{mode}_{dim}.pickle'.format(name=name, mode=emb_mode, dim=str(dim_n)))
    tmp_dict = {}
    for i, item in zip(tmp[tmp.columns[0]].values, tmp[tmp.columns[1:]].values):
        tmp_dict[i] = item
    return tmp_dict  # id_to_embedding 字典


def embedding_mat(feat_name, dim_n, emb_mode):
    model = read_id_dict(feat_name, dim_n, emb_mode)  # id_to_embedding 字典
    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")

    embed_matrix = np.zeros((len(id2index) + 1, dim_n))
    for word, i in id2index.items():  # 遍历 id_to_index 字典
        embedding_vector = model[word] if word in model else None  # 使用 id 取出 对应 embedding
        if embedding_vector is not None:
            embed_matrix[i] = embedding_vector  # 建立 index_to_embedding 矩阵
        else:
            unk_vec = np.random.random(dim_n) * 0.5  # 如果这个id没有预训练的embedding则随机初始化
            unk_vec = unk_vec - unk_vec.mean()
            embed_matrix[i] = unk_vec

    return embed_matrix


# 建立 index to embedding 字典
dim_n = 150
emb_m = "vec"
tfidf_svd_user_feed = embedding_mat("user_feed",dim_n,emb_m)
tfidf_svd_feed_user = embedding_mat("feed_user",dim_n,emb_m)
tfidf_svd_user_author = embedding_mat("user_author",dim_n,emb_m)
tfidf_svd_author_user = embedding_mat("author_user",dim_n,emb_m)


dim_n = 128
emb_m = "vec"
tfidf_svd_tag_user = embedding_mat("user_tag",32,emb_m)
tfidf_svd_tag_feed = embedding_mat("feed_tag",32,emb_m)

tfidf_svd_hkey_user = embedding_mat("user_key1",dim_n,emb_m)
tfidf_svd_mkey_user = embedding_mat("user_key2",dim_n,emb_m)
tfidf_svd_hkey_feed = embedding_mat("feed_key1",dim_n,emb_m)
tfidf_svd_mkey_feed = embedding_mat("feed_key2",dim_n,emb_m)

tfidf_svd_feed_emb = embedding_mat("feed_emb",512,emb_m)

dim_n = 150
emb_m = "d2v"
user_feed_d2v = embedding_mat("user_feed",dim_n,emb_m)
feed_user_d2v = embedding_mat("feed_user",dim_n,emb_m)
user_author_d2v = embedding_mat("user_author",dim_n,emb_m)
author_user_d2v = embedding_mat("author_user",dim_n,emb_m)

data_set = user_action[["userid","feedid"]+label_cols].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
data_set = data_set.merge(feed_info[["feedid","authorid","videoplayseconds"]], how='left',on="feedid")

# 设立新的标签 这个是创新点之一
data_set["play"] = data_set["play"] / 1000
data_set["play"] = data_set["play"] / data_set["videoplayseconds"]
data_set["play"] = data_set["play"].apply(lambda x:1 if x>0.9 else 0)
data_set = data_set.astype(int)


def user_graph(df):
    if not os.path.exists(features_path+"user_graph.pickle"):
        tmp = df[df.follow==1][["userid","authorid"]].drop_duplicates() # 取出所有follow的记录
        tmp['logic']=0.0
        tmp.to_pickle(features_path+"user_graph.pickle")


user_graph(data_set)


def pad_seq(df, feat_name, max_len=1, mode='data'):
    tmp = df[[feat_name]].copy() # 取出这一列

    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")

    tmp[feat_name] = tmp[feat_name].apply(lambda x:id2index[x]) # 将这一列的id改为index
    seq = pad_sequences(tmp.values, maxlen = max_len) # 填空值吗？结论：不可能有空值

    return seq

# 这四个的index是一一对应的关系 data_userid[0] data_feedid[0] data_authorid[0] data_labels[0] 对应第0条记录
# 且里面都是 index 而不是 id


data_userid = pad_seq(data_set, "userid", mode="data")
data_feedid = pad_seq(data_set, "feedid", mode="data")
data_authorid = pad_seq(data_set, "authorid", mode="data")
data_feats = np.concatenate([data_userid,data_feedid,data_authorid],axis=1) # 这里将他们组合
data_labels = data_set[label_cols].values.reshape(-1,8)

feature_length_list = [data_userid.max() + 1, data_feedid.max() + 1, data_authorid.max() + 1]
feature_length = 0
first_order_shifts = []
for length in feature_length_list:
    first_order_shifts.append(length)
    feature_length += length


name = "dota_m1v128k10"
if not os.path.exists(models_path + name):
    os.mkdir(models_path + name)
phase_feat = "semi"
epochs_ = 1
k_folds = 10
lr_rate = 1e-3
batch_size = 2048


# seed = 16
# 5 [64, 64, 64] 0.7279571428571429
# 5 [64, 64, 64] 0.7322857142857142 缩放点积
# 4 [64, 64, 64] 0.7341142857142857 缩放点积
# 4 [64, 64, 64] 0.7290857142857143
# 3 [64, 64, 64] 0.7295571428571429
# 3 [64, 64, 64] 0.7340428571428571 缩放点积
# 4 [64, 64, 64, 64]  0.7328428571428571 缩放点积
# 8 [64, 64, 64]   0.7341857142857143 缩放点积
# 7 [64, 64, 64]    缩放点积 0.7318142857142858
# 6 [64, 64, 64]   0.7353285714285713 缩放点积 winner
# 缩放点积稳定0.005的提升
num_shared_experts = 6
num_tasks = 8
experts_shape = [64, 64, 64]

kf = KFold(n_splits=k_folds, shuffle=True, random_state=410).split(data_feats)
for i, (train_fold, valid_fold) in enumerate(kf):
    print(i, 'fold')
    #    train_fold = shuffle(train_fold, random_state=2020)
    train_feats = data_feats[train_fold]
    train_labels = data_labels[train_fold]
    print(train_feats.shape)
    # =========================================== fit =================================================
    the_path = models_path + "{name}/{flag}_k{n}".format(name=name, flag=phase_feat, n=str(i))
    if not os.path.exists(the_path):
        os.mkdir(the_path)

    model = MMoE(
        tfidf_svd_user_feed, tfidf_svd_feed_user, tfidf_svd_user_author,
        tfidf_svd_author_user, tfidf_svd_feed_emb, tfidf_svd_tag_user,
        tfidf_svd_hkey_user, tfidf_svd_mkey_user, tfidf_svd_tag_feed,
        tfidf_svd_hkey_feed, tfidf_svd_mkey_feed, user_feed_d2v,
        feed_user_d2v, user_author_d2v, author_user_d2v, first_order_shifts,
        feature_length,

        num_shared_experts=num_shared_experts, num_tasks=num_tasks, experts_shape=experts_shape
    )
    checkpoint = ModelCheckpoint(the_path + '/model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    csv_logger = CSVLogger(the_path + '/log_{f}.log'.format(f=str(i)))
    model.compile(optimizer=tf.optimizers.Nadam(lr=lr_rate),
                  loss=[tf.keras.losses.binary_crossentropy for i in range(num_tasks)],
                  loss_weights=[1, 1, 1, 1, 1, 1, 1, 0.1])
    model.fit(train_feats[:],
              [train_labels[:, i] for i in range(num_tasks)],
              batch_size=batch_size,
              epochs=epochs_,
              callbacks=[checkpoint, csv_logger])
    gauc_list = get_gaucs(data_feats, data_labels, valid_fold, model)
    K.clear_session()
    break

res = np.array(gauc_list)
print(res.mean())
np.save(res_path+"origin_with_FM_res.npy",res)

