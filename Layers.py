import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import layers, Model
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.backend import expand_dims, repeat_elements, sum
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


class EmbeddingLayer(tf.keras.Model):
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
            **kwargs
    ):
        super(EmbeddingLayer, self).__init__(**kwargs)

        # Feature fields of tfidf+svd interactions and original embedding of feeds
        # tfidf + svd user to feed
        self.UserFeedEmbedding = Embedding(
            input_dim=tfidf_svd_user_feed.shape[0],
            output_dim=tfidf_svd_user_feed.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_user_feed),
            dtype='float32'
        )
        # tfidf + svd feed to user
        self.FeedUserEmbedding = Embedding(
            input_dim=tfidf_svd_feed_user.shape[0],
            output_dim=tfidf_svd_feed_user.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_feed_user),
            dtype='float32'
        )
        # tfidf + svd user to author
        self.UserAuthorEmbedding = Embedding(
            input_dim=tfidf_svd_user_author.shape[0],
            output_dim=tfidf_svd_user_author.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_user_author),
            dtype='float32'
        )
        # tfidf + svd author to user
        self.AuthorUserEmbedding = Embedding(
            input_dim=tfidf_svd_author_user.shape[0],
            output_dim=tfidf_svd_author_user.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_author_user),
            dtype='float32'
        )
        # tfidf + svd feed embedding
        self.FeedEmbEmbedding = Embedding(
            input_dim=tfidf_svd_feed_emb.shape[0],
            output_dim=tfidf_svd_feed_emb.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_feed_emb),
            trainable=False, dtype='float32'
        )


        # Feature fields for tfidf + svd: tag and keywords
        self.UserTagEmbedding = Embedding(
            input_dim=tfidf_svd_tag_user.shape[0],
            output_dim=tfidf_svd_tag_user.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_tag_user),
            dtype='float32'
        )
        self.UserKey1Embedding = Embedding(
            input_dim=tfidf_svd_hkey_user.shape[0],
            output_dim=tfidf_svd_hkey_user.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_hkey_user),
            dtype='float32'
        )
        self.UserKey2Embedding = Embedding(
            input_dim=tfidf_svd_mkey_user.shape[0],
            output_dim=tfidf_svd_mkey_user.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_mkey_user),
            dtype='float32'
        )
        self.FeedTagEmbedding = Embedding(
            input_dim=tfidf_svd_tag_feed.shape[0],
            output_dim=tfidf_svd_tag_feed.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_tag_feed),
            dtype='float32'
        )
        self.FeedKey1Embedding = Embedding(
            input_dim=tfidf_svd_hkey_feed.shape[0],
            output_dim=tfidf_svd_hkey_feed.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_hkey_feed),
            dtype='float32'
        )
        self.FeedKey2Embedding = Embedding(
            input_dim=tfidf_svd_mkey_feed.shape[0],
            output_dim=tfidf_svd_mkey_feed.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(tfidf_svd_mkey_feed),
            dtype='float32'
        )

        # Feature fields for Doc2Vec
        self.UserFeedD2vEmbedding = Embedding(
            input_dim=user_feed_d2v.shape[0],
            output_dim=user_feed_d2v.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(user_feed_d2v),
            dtype='float32'
        )
        self.FeedUserD2vEmbedding = Embedding(
            input_dim=feed_user_d2v.shape[0],
            output_dim=feed_user_d2v.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(feed_user_d2v),
            dtype='float32'
        )
        self.UserAuthorD2vEmbedding = Embedding(
            input_dim=user_author_d2v.shape[0],
            output_dim=user_author_d2v.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(user_author_d2v),
            dtype='float32'
        )

        self.AuthorUserD2vEmbedding = Embedding(
            input_dim=author_user_d2v.shape[0],
            output_dim=author_user_d2v.shape[1],
            embeddings_initializer=tf.keras.initializers.constant(author_user_d2v),
            dtype='float32'
        )
        self.fist_order_shifts = first_order_shifts
        # feature transformation layers
        self.Dense256 = Dense(256, activation='relu')
        self.Dense128 = Dense(128, activation='relu')
        self.Dropout3 = Dropout(0.3)
        self.Dropout2 = Dropout(0.2)

    def call(self, inputs):
        inputs = tf.reshape(inputs, shape=[-1, 3])
        # ont_hot_inputs for fm
        one_hot_inputs = inputs + tf.convert_to_tensor(self.fist_order_shifts)
        # get tfidf + svd interactions embeddings
        user_feed_embed = self.UserFeedEmbedding(inputs[:, 0])
        user_author_embed = self.UserAuthorEmbedding(inputs[:, 0])
        author_user_embed = self.AuthorUserEmbedding(inputs[:, 2])
        feed_emb_embed = self.FeedEmbEmbedding(inputs[:, 1])
        feed_user_embed = self.FeedUserEmbedding(inputs[:, 1])
        # get tfidf + svd tag and keywords embeddings
        user_tag_embed = self.UserTagEmbedding(inputs[:, 0])
        user_key1_embed = self.UserKey1Embedding(inputs[:, 0])
        user_key2_embed = self.UserKey2Embedding(inputs[:, 0])
        feed_tag_embed = self.FeedTagEmbedding(inputs[:, 1])
        feed_key1_embed = self.FeedKey1Embedding(inputs[:, 1])
        feed_key2_embed = self.FeedKey2Embedding(inputs[:, 1])
        # get Doc2Vec embeddings
        user_feed_d2vem = self.UserFeedD2vEmbedding(inputs[:, 0])
        user_author_d2vem = self.UserAuthorD2vEmbedding(inputs[:, 0])
        feed_user_d2vem = self.FeedUserD2vEmbedding(inputs[:, 1])
        author_user_d2vem = self.AuthorUserD2vEmbedding(inputs[:, 2])
        # feature interactions
        user_w_feed = tf.matmul(
            tf.expand_dims(user_feed_d2vem, -1),
            tf.expand_dims(feed_user_d2vem, -1),
            transpose_b=True
        )
        user_w_auth = tf.matmul(
            tf.expand_dims(user_author_d2vem, -1),
            tf.expand_dims(author_user_d2vem, -1),
            transpose_b=True
        )
        user_w_user = tf.concat([user_w_feed, user_w_auth], axis=1)
        user_w_user = tf.reshape(user_w_user, shape=[-1, user_w_user.shape[1] * user_w_user.shape[2]])
        user_w_user = self.Dropout3(self.Dense128(user_w_user))

        user_x_feed = tf.matmul(tf.expand_dims(user_feed_embed, -1), tf.expand_dims(feed_user_embed, -1),
                                transpose_b=True)
        user_x_auth = tf.matmul(tf.expand_dims(user_author_embed, -1), tf.expand_dims(author_user_embed, -1),
                                transpose_b=True)
        user_x_user = tf.concat([user_x_feed, user_x_auth], axis=1)
        user_x_user = tf.reshape(user_x_user, shape=[-1, user_x_user.shape[1] * user_x_user.shape[2]])
        user_x_user = self.Dropout2(self.Dense256(user_x_user))

        # Concatenate input for FM
        fm_feature_list = [user_feed_embed, user_author_embed, feed_user_embed,
                           author_user_embed,
                           ]
        embeds_fm = tf.concat([fm_feature_list], axis=1)
        fm_input = {'sparse_inputs': one_hot_inputs,
                    'embed_inputs': tf.reshape(embeds_fm, shape=(-1, len(fm_feature_list), 150))}

        # Concatenate input for experts
        expert_inputs = tf.concat(
            [user_feed_embed, user_author_embed, user_tag_embed,
             user_key1_embed, user_key2_embed, user_feed_d2vem,
             user_author_d2vem, feed_user_embed, feed_tag_embed,
             feed_key1_embed, feed_emb_embed, feed_key2_embed,
             feed_user_d2vem, author_user_embed, author_user_d2vem,
             user_x_user, user_w_user], axis=-1)

        return expert_inputs, fm_input


class MultiTask(tf.keras.Model):
    def __init__(self, num_shared_experts, num_tasks, experts_shape=[128], task_shape=None,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation=None,
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MultiTask, self).__init__(**kwargs)
        self.experts_shape = experts_shape
        self.task_shape = task_shape
        self.num_shared_experts = num_shared_experts
        self.num_tasks = num_tasks
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.shared_experts = []
        self.gate_layers = []
        self.task_specific_experts = []

        for i in range(self.num_shared_experts):
            self.shared_experts.append( # can use other models as experts
                [
                    layers.Dense(unit,
                                 activation=self.expert_activation,
                                 use_bias=self.use_expert_bias,
                                 kernel_initializer=self.expert_kernel_initializer,
                                 bias_initializer=self.expert_bias_initializer,
                                 kernel_regularizer=self.expert_kernel_regularizer,
                                 bias_regularizer=self.expert_bias_regularizer,
                                 activity_regularizer=None,
                                 kernel_constraint=self.expert_kernel_constraint,
                                 bias_constraint=self.expert_bias_constraint)
                    for unit in self.experts_shape
                ]
            )

        for i in range(self.num_tasks):
            self.gate_layers.append(layers.Dense(self.num_shared_experts + 1,
                                                 activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer,
                                                 activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))
            self.task_specific_experts.append(  # can use other models as experts
                [
                    layers.Dense(unit,
                                 activation=self.expert_activation,
                                 use_bias=self.use_expert_bias,
                                 kernel_initializer=self.expert_kernel_initializer,
                                 bias_initializer=self.expert_bias_initializer,
                                 kernel_regularizer=self.expert_kernel_regularizer,
                                 bias_regularizer=self.expert_bias_regularizer,
                                 activity_regularizer=None,
                                 kernel_constraint=self.expert_kernel_constraint,
                                 bias_constraint=self.expert_bias_constraint)
                    for unit in self.task_shape
                ]
            )


    def call(self, inputs):
        task_specific_experts_outputs, shared_experts_outputs, gate_outputs, final_outputs = [], [], [], []

        tmp = inputs
        for expert_layer in self.shared_experts:
            for dense in expert_layer:
                tmp = dense(tmp)
            expert_output = tf.expand_dims(tmp, axis=2)
            shared_experts_outputs.append(expert_output)
        shared_experts_info = tf.concat(shared_experts_outputs, 2)
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        tmp = inputs
        for expert_layer in self.task_specific_experts:
            for dense in expert_layer:
                tmp = dense(tmp)
            expert_output = tf.expand_dims(tmp, axis=2)
            task_specific_experts_outputs.append(expert_output)

        scaler = tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))

        for task, gate_output in enumerate(gate_outputs):
            gate_output = tf.nn.softmax(gate_output / scaler)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            specific_expert_info = task_specific_experts_outputs[task]
            candidate_info = tf.concat([specific_expert_info, shared_experts_info], 2)
            weighted_expert_output = candidate_info * repeat_elements(
                expanded_gate_output, self.experts_shape[-1], axis=1
            )
            final_outputs.append(sum(weighted_expert_output, axis=2))
        return final_outputs


class FM(Layer):
    def __init__(self, feature_length):
        super(FM, self).__init__()
        self.feature_length = feature_length

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),  # 特征总共有多少维就有多少个
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, **kwargs):
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']

        first_order = tf.nn.embedding_lookup(self.w, sparse_inputs)
        first_order = tf.reduce_sum(first_order, axis=2)

        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1)  # (batch_size, 1, embed_dim)

        second_order = 0.5 * tf.subtract(square_sum, sum_square)  # (batch_size, 1)

        out = tf.concat([first_order, second_order], axis=-1)

        return out
