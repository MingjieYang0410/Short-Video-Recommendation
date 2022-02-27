from layers import *


class CGC(Model):
    def __init__(
            self,
            tfidf_svd_user_feed, tfidf_svd_feed_user,
            tfidf_svd_user_author, tfidf_svd_author_user,
            tfidf_svd_feed_emb, tfidf_svd_tag_user,
            tfidf_svd_hkey_user, tfidf_svd_mkey_user,
            tfidf_svd_tag_feed, tfidf_svd_hkey_feed,
            tfidf_svd_mkey_feed, user_feed_d2v,
            feed_user_d2v, user_author_d2v,
            author_user_d2v, first_order_shifts,
            feature_length, num_shared_experts=4, num_tasks=8,
            experts_shape=[128], task_shape=None,
            **kwargs
    ):
        super(CGC, self).__init__(**kwargs)

        self.num_tasks = num_tasks
        self.embedding_layer = EmbeddingLayer(
            tfidf_svd_user_feed, tfidf_svd_feed_user,
            tfidf_svd_user_author, tfidf_svd_author_user,
            tfidf_svd_feed_emb, tfidf_svd_tag_user,
            tfidf_svd_hkey_user, tfidf_svd_mkey_user,
            tfidf_svd_tag_feed, tfidf_svd_hkey_feed,
            tfidf_svd_mkey_feed, user_feed_d2v,
            feed_user_d2v, user_author_d2v,
            author_user_d2v, first_order_shifts,
        )

        self.MTLayer = MultiTask(
            num_shared_experts=num_shared_experts, num_tasks=num_tasks,
            experts_shape=experts_shape, task_shape=task_shape
        )
        self.Factorization_Machines = []
        self.MTTowerDense1 = []
        self.MTDropouts1 = []
        self.MTTowerDense2 = []
        self.MTDenseSigs = []

        # ['read_comment','like','click_avatar','forward','favorite','comment','follow']
        for i in range(self.num_tasks):
            self.Factorization_Machines.append(FM(feature_length))
            self.MTTowerDense1.append(Dense(128, activation=PReLU()))
            self.MTDropouts1.append(Dropout(0.2))
            self.MTTowerDense2.append(Dense(128, activation=PReLU()))
            self.MTDenseSigs.append(Dense(1, activation='sigmoid'))

    def call(self, inputs):
        expert_inputs, fm_inputs = self.embedding_layer(inputs)
        dmt = self.MTLayer(expert_inputs)

        outputs = []
        fm_outs = []
        for i in range(self.num_tasks):
            fm_out = self.Factorization_Machines[i](fm_inputs)
            fm_outs.append(fm_out)

        for task, action in enumerate(dmt):
            x = self.MTDropouts1[task](self.MTTowerDense1[task](action))
            x = self.MTTowerDense2[task](x)
            cat_x = tf.concat([x, fm_outs[task]], axis=1)
            x = self.MTDenseSigs[task](cat_x)
            outputs.append(x)

        return outputs  # tf.concat(outputs,axis=-1)

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)



