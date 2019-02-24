import tensorflow as tf
import tensorflow_hub as hub


def euclidean_dist(emb1, emb2):
    """ Calculates the euclidean distance between all samples in emb1 and emb2. """
    # Get a square matrix of all combinations of a - b
    diffs = tf.expand_dims(emb1, axis=1) - tf.expand_dims(emb2, axis=0)
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1))


def top_model(bottleneck, embed_dim, is_training=True):
    with tf.variable_scope('top', reuse=tf.AUTO_REUSE):
        if is_training:
            bottleneck = tf.nn.dropout(bottleneck, keep_prob=0.5)

        emb_x1 = tf.layers.dense(bottleneck, embed_dim, activation=tf.nn.relu,
                                 kernel_regularizer=tf.nn.l2_loss, name='fc1',
                                 reuse=tf.AUTO_REUSE)

    return emb_x1, emb_x1


def encoder(x, embedding_network, embed_dim, is_training=True):
    with tf.variable_scope('base_network', reuse=tf.AUTO_REUSE):
        bottleneck = embedding_network(x)

    emb_x1, emb_x2 = top_model(bottleneck, embed_dim, is_training)
    return bottleneck, emb_x1, emb_x2


def get_centroids(embeddings, labels, n_way):
    labels_one_hot = tf.one_hot(labels, depth=n_way)
    return tf.map_fn(lambda x: tf.reduce_mean(tf.boolean_mask(embeddings, x), axis=0),
                     tf.transpose(labels_one_hot), tf.float32)


class SemiHardModel(object):

    def __init__(self, config, train_top_only=False):
        n_way = config['n_way']
        embed_dim = config['embed_dim']
        image_size = config['image_size']
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('global_step'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope('training'):
                with tf.name_scope('input_placeholder'):
                    self.image_ph = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='image')
                    self.bottleneck_ph = tf.placeholder(tf.float32, shape=(None, 2048), name='bottleneck')
                    self.label_ph = tf.placeholder(tf.int64, shape=(None, ), name='label')
                    self.centroid_ph = tf.placeholder(tf.float32, shape=(n_way, embed_dim), name='centroid')

                if train_top_only:
                    with tf.name_scope('embedding_fn'):
                        emb_x1, emb_x2 = top_model(self.bottleneck_ph, embed_dim)
                        self.bottleneck = self.bottleneck_ph
                else:
                    embedding_network_train = hub.Module(config['module_spec'], trainable=True, tags={'train'})
                    with tf.name_scope('embedding_fn'):
                        self.bottleneck, emb_x1, emb_x2 = \
                            encoder(self.image_ph, embedding_network_train, embed_dim, is_training=True)

                with tf.name_scope('centroid_out'):
                    self.centroids = get_centroids(emb_x2, self.label_ph, n_way)

                with tf.name_scope('siamese_bh_loss'):
                    reg_terms = tf.losses.get_regularization_losses()
                    self.bh_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
                        labels=self.label_ph,
                        embeddings=emb_x2,
                        margin=config['tl_margin']
                    )
                    self.loss_summary = tf.summary.scalar('batch_hard_loss', self.bh_loss)
                    self.total_loss = config['lambda_reg'] * tf.reduce_sum(reg_terms) + self.bh_loss
                    self.total_loss_summary = tf.summary.scalar('total_loss_with_reg', self.total_loss)

                with tf.name_scope('metrics/ncc'):
                    self.dists_qc = euclidean_dist(emb_x2, self.centroid_ph)
                    self.pred = tf.argmin(self.dists_qc, axis=-1)
                    self.acc = tf.reduce_mean(tf.cast(tf.equal(self.label_ph, self.pred), tf.float32))
                    self.acc_summary = tf.summary.scalar('accuracy', self.acc)

                    self.pred5 = tf.to_int64(tf.nn.top_k(-self.dists_qc, k=5, sorted=True).indices)
                    self.accs = tf.map_fn(lambda p: tf.reduce_mean(tf.to_float(tf.equal(p, self.label_ph))),
                                          tf.transpose(self.pred5), dtype=tf.float32)
                    self.acc_top5 = tf.reduce_sum(self.accs)
                    self.acc_top5_summary = tf.summary.scalar('accuracy_top5', self.acc_top5)

                with tf.name_scope('training_op'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
                    vars_to_train = tf.trainable_variables()
                    if not train_top_only:
                        vars_to_train = vars_to_train[config['finetune_layers_after']:]

                    self.training_op = optimizer.minimize(self.total_loss, var_list=vars_to_train,
                                                          global_step=self.global_step)

            with tf.name_scope('eval'):
                with tf.name_scope('input_placeholder'):
                    image_ph_eval = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='image')
                    bottleneck_ph_eval = tf.placeholder(tf.float32, shape=(None, 2048), name='bottleneck_eval')
                    label_ph_eval = tf.placeholder(tf.int64, shape=(None, ), name='label')
                    centroid_ph_eval = tf.placeholder(tf.float32, shape=(None, embed_dim), name='centroid')

                if train_top_only:
                    with tf.name_scope('embedding_fn'):
                        emb_x1_eval, emb_x2_eval = top_model(bottleneck_ph_eval, embed_dim, is_training=False)
                        self.bottleneck_eval = bottleneck_ph_eval
                else:
                    embedding_network_eval = hub.Module(config['module_spec'], trainable=False)
                    with tf.name_scope('embedding_fn'):
                        bottleneck_eval, emb_x1_eval, emb_x2_eval = \
                            encoder(image_ph_eval, embedding_network_eval, is_training=False)

                with tf.name_scope('centroid_out'):
                    self.centroids_eval = get_centroids(emb_x2_eval, label_ph_eval, n_way)

                with tf.name_scope('siamese_bh_loss'):
                    self.bh_loss_eval = tf.contrib.losses.metric_learning.triplet_semihard_loss(
                        labels=label_ph_eval,
                        embeddings=emb_x2_eval,
                        margin=config['tl_margin']
                    )
                    self.loss_eval_summary = tf.summary.scalar('batch_hard_loss', self.bh_loss_eval)

                with tf.name_scope('metrics/ncc'):
                    self.dists_qc_eval = euclidean_dist(emb_x2_eval, centroid_ph_eval)
                    self.pred_eval = tf.argmax(-self.dists_qc_eval, axis=-1)
                    self.acc_eval = tf.reduce_mean(tf.cast(tf.equal(label_ph_eval, self.pred_eval), tf.float32))
                    self.acc_eval_summary = tf.summary.scalar('accuracy', self.acc_eval)

                    self.pred5_eval = tf.to_int64(tf.nn.top_k(-self.dists_qc_eval, k=5, sorted=True).indices)
                    self.accs_eval = tf.map_fn(lambda p: tf.reduce_mean(tf.to_float(tf.equal(p, label_ph_eval))),
                                               tf.transpose(self.pred5_eval), dtype=tf.float32)
                    self.acc_top5_eval = tf.reduce_sum(self.accs_eval)
                    self.acc_top5_eval_summary = tf.summary.scalar('acc_top5', self.acc_top5_eval)

            with tf.name_scope('init'):
                init_op_global = tf.global_variables_initializer()
                init_op_local = tf.local_variables_initializer()

            self.init = [init_op_global, init_op_local]

            with tf.name_scope('save'):
                self.saver = tf.train.Saver(name='checkpoint')
                exclusions = ['module']
                vars_to_restore = []
                for var in tf.global_variables():
                    for exclusion in exclusions:
                        if var.op.name.startswith(exclusion):
                            break
                    else:
                        vars_to_restore.append(var)

                self.saver_top = tf.train.Saver(name='checkpoint', var_list=vars_to_restore)
