from collections import OrderedDict
from itertools import chain
import json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import tensorflow as tf
import yaml


def get_bottleneck_config(bottleneck_dir='data/modems_bottleneck_v3', splits_dir='data/modem_split', external_dir=None):
    """
    Checks that we have the correct bottleneck features, and that the cached
    features on disk has the same directory hierarchy.

    Data preparation steps:

    1. Split the dataset
    2. Generate the bottleneck features and cache them on disk.

    Features are organized using the same hierarchy as the split image sets.

    :param bottleneck_dir:
    :param splits_dir:
    :param external_dir: external dataset (other device images)
    :return:
    """
    with open(os.path.join(bottleneck_dir, 'meta.json'), 'r') as f:
        meta_bottleneck = json.load(f)

    with open(os.path.join(splits_dir, 'meta.json'), 'r') as f:
        meta_images = json.load(f)

    is_consistent = (meta_bottleneck['list']['train'] == meta_images['list']['train'] and
                     meta_bottleneck['list']['val'] == meta_images['list']['val'] and
                     meta_bottleneck['list']['test'] == meta_images['list']['test'])
    external_consistency = False
    if external_dir:
        if os.path.exists(os.path.join(bottleneck_dir, 'external')):
            files = list(os.walk(external_dir))[1:]
            external = list(map(lambda x: (x[0].split('/')[-1],
                                           list(map(lambda y: os.path.join(x[0].split('/')[-1], y), x[2]))), files))
            external = OrderedDict(external)
            if 'external' in meta_bottleneck['list'] and 'external' in meta_images['list']:
                c1 = meta_bottleneck['list']['external'] == external
                c2 = meta_images['list']['external'] == external
                external_consistency = c1 and c2
    else:
        external_consistency = True

    if external_dir and not external_consistency:
        files = list(os.walk(external_dir))[1:]
        external = list(map(lambda x: (x[0].split('/')[-1],
                                       list(map(lambda y: os.path.join(x[0].split('/')[-1], y), x[2]))), files))
        meta_images['list']['external'] = OrderedDict(external)
        meta_images['split']['external'] = external_dir
        with open(os.path.join(splits_dir, 'meta.json'), 'w') as f:
            json.dump(meta_images, f, indent=4)

    return is_consistent, external_consistency, meta_bottleneck


def get_dataset_info(splits_dir):
    if not os.path.exists(splits_dir):
        raise Exception('{} not found. Run `split_dataset` first.'.format(os.path.join(os.getcwd(), splits_dir)))

    with open(os.path.join(splits_dir, 'meta.json'), 'r') as f:
        return json.load(f)


def load_hyperparams(file_path):
    with open(file_path) as f:
        return yaml.load(f)


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if value is not None:
            merged[key] = value

    return merged


def run_training(model, config):
    top_only_tb_writer = None
    train_tb_writer = None
    val_tb_writer = None
    top_val_tb_writer = None
    train_top_only = config['train_top_only']
    top_model_val_dir = config['top_model_val_dir']
    top_model_dir = config['top_model_dir']
    train_dir = config['train_dir']
    if train_top_only:
        top_only_tb_writer = tf.summary.FileWriter(top_model_dir, model.graph)
        top_val_tb_writer = tf.summary.FileWriter(top_model_val_dir, model.graph)
    else:
        train_tb_writer = tf.summary.FileWriter(train_dir, model.graph)
        val_tb_writer = tf.summary.FileWriter(config['val_dir'], model.graph)

    with tf.Session(graph=model.graph) as sess:
        sess.run(model.init)
        checkpoint_path_primary = tf.train.latest_checkpoint(train_dir)
        checkpoint_path_top_only = tf.train.latest_checkpoint(top_model_dir)
        if train_top_only:
            if checkpoint_path_top_only:
                model.saver_top.restore(sess, checkpoint_path_top_only)
                print('Restored top model from top model checkpoint')
        else:
            if checkpoint_path_primary:
                model.saver.restore(sess, checkpoint_path_primary)
                print('Restored full model from primary checkpoint')
            elif checkpoint_path_top_only:
                model.saver_top.restore(sess, checkpoint_path_top_only)
                print('Partially restored full model from top model checkpoint')

        global_step = sess.run(model.global_step)
        data_gen = config['data_flow_gen']
        image_size = config['image_size']
        for epoch in range(config['n_epochs']):
            data = data_gen.next_block_batch(split='train', target_size=(image_size, image_size),
                                             img_gen=config['generator'], n_way=config['n_way'],
                                             k_sample=config['k_sample'], clz_to_sample=None)
            x_sample = data['x_sample']
            y_sample = data['y_sample']
            data_feed = np.concatenate(x_sample)
            data_feed = data_feed.astype(np.float32)
            label_feed = np.concatenate(y_sample)
            label_feed = label_feed.astype(np.int32)
            if config['train_top_only']:
                feed_dict_train = {
                    model.bottleneck_ph: data_feed,
                    model.label_ph: label_feed
                }
            else:
                feed_dict_train = {
                    model.image_ph: data_feed,
                    model.label_ph: label_feed
                }

            values = sess.run([model.global_step, model.bh_loss, model.training_op,
                               model.centroids, model.total_loss, model.loss_summary,
                               model.total_loss_summary], feed_dict=feed_dict_train)
            global_step = values[0]
            support_centroids = values[3]
            print(values[:3], '>>>>> Training >>>>>', global_step)
            print('total loss:', values[4])

            if train_tb_writer:
                for i in range(1, 3):
                    train_tb_writer.add_summary(values[-i], global_step)

            if top_only_tb_writer:
                for i in range(1, 3):
                    top_only_tb_writer.add_summary(values[-i], global_step)

            # Evaluate every n steps
            if global_step % config['eval_every_n_steps'] == 0:
                data = data_gen.next_block_batch(split='val', target_size=(image_size, image_size),
                                                 n_way=config['n_way'], k_sample=config['k_sample'],
                                                 clz_to_sample=None)
                x_sample = data['x_sample']
                y_sample = data['y_sample']
                data_feed = np.concatenate(x_sample)
                data_feed = data_feed.astype(np.float32)
                label_feed = np.concatenate(y_sample)
                label_feed = label_feed.astype(np.int32)
                if config['train_top_only']:
                    feed_dict_eval = {
                        model.bottleneck_ph: data_feed,
                        model.label_ph: label_feed,
                        model.centroid_ph: support_centroids
                    }
                else:
                    feed_dict_eval = {
                        model.image_ph: data_feed,
                        model.label_ph: label_feed,
                        model.centroid_ph: support_centroids
                    }

                values = sess.run([model.bh_loss, model.acc, model.acc_top5, model.pred, model.pred5,
                                   model.accs, model.dists_qc, model.loss_summary, model.acc_summary,
                                   model.acc_top5_summary, model.total_loss_summary], feed_dict=feed_dict_eval)
                print('accs:', values[5])
                print('val loss: {}, val acc: {}, val top5_acc: {}\n'.format(values[0], values[1], values[2]))

                if val_tb_writer:
                    for i in range(1, 3):
                        val_tb_writer.add_summary(values[-i], global_step)

                if top_val_tb_writer:
                    for i in range(1, 3):
                        top_val_tb_writer.add_summary(values[-i], global_step)

        if not train_top_only:
            model.saver.save(sess, os.path.join(train_dir, 'model-' + str(global_step) + '.ckpt'))
        else:
            model.saver.save(sess, os.path.join(top_model_dir, 'model-' + str(global_step) + '.ckpt'))


class ImageFlowGenerator(object):
    """
    A data generator to meet the requirements of different models.

    1. Softmax + cross-entropy:

       Requires random batch of training data. Use `get_random_batch`

    2. (Semi-)hard batch

       Requires to sample `k_sample` samples from each of `n_way` classes.
       Use method `next_block_batch`.

    3. Prototypical net

       Requires to sample `n_support` samples and another `n_query` samples
       from `n_way` classes. Use `next_meta_batch`.
    """

    def __init__(self, config, mode='image'):
        if mode == 'image':
            self.splits = config['split']
            self.loader = load_image_as_array
        elif mode == 'bottleneck':
            self.splits = config['bottleneck_split']
            self.loader = _load_bottleneck_features
        else:
            raise ValueError('Invalid mode: "{}"'.format(mode))

        self.lists = config['list']
        self.n_way_training = len(self.lists['train'])
        if 'external' in self.lists:
            self.n_way_external = len(self.lists['external'])
        else:
            self.n_way_external = 0

        self.sample_sizes = OrderedDict({
            'train': sum(list(map(lambda k: len(k[1]), self.lists['train'].items()))),
            'val': sum(list(map(lambda k: len(k[1]), self.lists['val'].items()))),
            'test': sum(list(map(lambda k: len(k[1]), self.lists['test'].items())))
        })
        self.n_classes_train = len(self.lists['train']) - 1
        self.n_classes_val = len(self.lists['val']) - 1
        self.n_classes_test = len(self.lists['test']) - 1
        self.img_gen = ImageDataGenerator(rescale=1/255)
        self.random_flow_gen_register = OrderedDict()

    # def meta_info_builder(self):
    #     for split in ['train', 'val', 'test', 'minor', 'rare']:
    #         sample_size = 0
    #         sample_sizes = []
    #         a = {
    #             'split': {
    #                 'sample_size': sample_size,
    #                 'sample_sizes': sample_sizes,
    #                 'dir': 0
    #             }
    #         }

    def _meta_info(self, split):
        if isinstance(split, str):
            split_dir = self.splits[split]
            clz_files = list(map(lambda x: (x[0], list(map(lambda y: os.path.join(split_dir, y), x[1]))),
                                 self.lists[split].items()))
        elif isinstance(split, (list, tuple)):
            assert len(split) == 2
            clz_files = self._merge_clz_files(split[0], split[1]).items()
        else:
            raise ValueError()

        labels = []
        sample_sizes = []
        data = OrderedDict()  # by class
        idx_label_global = []
        for i, clz_flist in enumerate(clz_files):
            label = clz_flist[0]
            labels.append(label)
            flist = clz_flist[1]
            sample_size = len(flist)
            sample_sizes.append(sample_size)
            data[label] = (flist, [i] * sample_size)
            idx_label_global.append(label)

        return data, idx_label_global

    def get_random_batch(self, split, target_size=(299, 299), batch_size=32,
                         img_gen=None, one_round=False, shuffle=False):
        if isinstance(split, (list, tuple)):
            assert len(split) == 2

        settings = (tuple(split), target_size, batch_size, img_gen, one_round, shuffle)
        if settings in self.random_flow_gen_register.keys():
            rf = self.random_flow_gen_register[settings]
        else:
            rf = self.random_flow(split, target_size=target_size, batch_size=batch_size,
                                  img_gen=img_gen, one_round=one_round, shuffle=shuffle)
            self.random_flow_gen_register[settings] = rf

        data_feed = next(rf)
        if data_feed['finished']:
            del self.random_flow_gen_register[settings]

        return data_feed

    def random_flow(self, split, target_size=(299, 299), batch_size=32,
                    img_gen=None, one_round=False, shuffle=False):
        """

        :param split: one of ['train', 'val', 'test']
        :param target_size:
        :param batch_size:
        :param img_gen:
        :param one_round: if False, generate data indefinitely
        :param shuffle:
        :return:
        """
        img_gen = img_gen or self.img_gen
        data, idx_label_global = self._meta_info(split)
        list_with_labels = list(map(lambda x: list(zip(*x)), data.values()))
        flist_all = list(chain(*list_with_labels))
        total_sample_size = len(flist_all)
        clz_to_sample = np.arange(len(list_with_labels))
        indices = np.arange(total_sample_size)
        if shuffle:
            np.random.shuffle(indices)

        i = 0
        while True:
            while True:
                if one_round and i >= total_sample_size:
                    yield None
                else:
                    break

            if batch_size > 0 and not one_round:
                sample_indices = indices[i:i + batch_size]
                i += batch_size
                if i > total_sample_size:
                    if shuffle:
                        np.random.shuffle(indices)

                    print(type(sample_indices))
                    sample_indices = np.concatenate([sample_indices, indices[0:i - total_sample_size]])
                    i -= total_sample_size
            elif batch_size > 0 and one_round:
                sample_indices = indices[i:i + batch_size]
                i += batch_size
            elif batch_size == -1:
                sample_indices = indices
            else:
                raise Exception('batch_size should be gt 0 or eq -1')

            x_sample = list(map(lambda x: self.loader(img_path=flist_all[x][0],
                                                      img_gen=img_gen,
                                                      target_size=target_size), sample_indices))
            y_sample = list(map(lambda x: flist_all[x][0], sample_indices))
            sample_path = list(map(lambda x: flist_all[i][0], sample_indices))
            if one_round and i >= total_sample_size:
                print(total_sample_size, '/', total_sample_size, '--- random flow')
            else:
                print(i, '/', total_sample_size, '--- random flow')

            yield {
                'x_sample': x_sample,
                'y_sample': y_sample,
                'sample_path': sample_path,
                'sample_indices': sample_indices,
                'idx_label': idx_label_global,
                'sample_clz': clz_to_sample,
                'finished': i >= total_sample_size
            }

    def _merge_clz_files(self, split1, split2):
        dir1 = self.splits[split1]
        dir2 = self.splits[split2]
        flist1 = list(map(lambda x: (x[0], list(map(lambda y: os.path.join(dir1, y), x[1]))),
                          self.lists[split1].items()))
        flist2 = list(map(lambda x: (x[0], list(map(lambda y: os.path.join(dir2, y), x[1]))),
                          self.lists[split2].items()))
        dic1 = OrderedDict(flist1)
        dic2 = OrderedDict(flist2)
        dic_merged = OrderedDict(list(dic1.items()))
        for k in dic2:
            if k in dic_merged:
                dic_merged[k] += dic2[k]
            else:
                dic_merged[k] = dic2[k]

        return dic_merged

    def next_block_batch(self, split, target_size, img_gen=None, n_way=None, k_sample=None, clz_to_sample=None):
        """
        Default setting gives the whole training set.

        :param split: one of 'train', 'val', 'test'
        :param target_size: InceptionV3 default is 299
        :param img_gen: (keras.preprocessing.image.ImageDataGenerator)
        :param n_way:
        :param k_sample:
        :param clz_to_sample: list of class indices
        :return:
        """
        img_gen = img_gen or self.img_gen
        if split not in ['train', 'val', 'test']:
            raise ValueError('`split` must be one of "train", "val" or "test"')

        if isinstance(split, str):
            split_dir = self.splits[split]
            clz_files = list(map(lambda x: (x[0], list(map(lambda y: os.path.join(split_dir, y), x[1]))),
                                 self.lists[split].items()))
        elif isinstance(split, (list, tuple)):
            assert len(split) == 2
            clz_files = self._merge_clz_files(split[0], split[1].items())
        else:
            raise ValueError()

        if n_way is None and clz_to_sample is None:
            clz_to_sample = np.arange(len(clz_files))
        elif n_way is not None and clz_to_sample is None:
            clz_to_sample = np.arange(len(clz_files))
            np.random.shuffle(clz_to_sample)
            clz_to_sample = clz_to_sample[:n_way]

        support_set_x = []
        support_set_y = []
        idx_label = []
        n_sampled = k_sample
        for i, j in enumerate(clz_to_sample):
            if j < len(clz_files):
                if k_sample is not None:
                    try:
                        img_sampled = np.random.choice(clz_files[j][1], size=k_sample, replace=False)
                    except ValueError:
                        img_sampled = np.random.choice(clz_files[j][1], size=k_sample, replace=True)
                else:
                    # using all available images
                    img_sampled = clz_files[j][1]
                    n_sampled = len(img_sampled)

                # add support set
                s = list(map(lambda fn: self.loader(img_path=fn, img_gen=img_gen, target_size=target_size),
                             img_sampled))
                support_set_x.append(np.stack(s))
                support_set_y.append(i * np.ones((n_sampled, )))
                idx_label.append(clz_files[j][0])

        # eventually get several 5D tensors [n_way, sample size for each class, width, height, channel]

        return {
            'x_sample': support_set_x,
            'y_sample': support_set_y,
            'idx_label': idx_label,
            'sample_clz': clz_to_sample
        }

    def next_metaset_batch(self, split, target_size, img_gen=None, clz_to_sample=None,
                           n_support=None, n_query=None, n_way=None):
        """
        Generate input for prototypical network.

        If both `n_way` and `clz_to_sample` is None, then sample all classes.

        :param split:
        :param target_size: target input size
        :param img_gen: (keras.preprocessing.image.ImageDataGenerator) used
               to specify data augmentation settings
        :param clz_to_sample: list of class indices, which can be obtained
               by `sample_clz` key in some return value
        :param n_support: size of support set
        :param n_query: size of query set
        :param n_way: number of classes to sample
        :return:
        """
        assert n_support is not None
        assert n_query is not None
        img_gen = img_gen or self.img_gen
        if split not in ['train', 'val', 'test']:
            raise ValueError('`split` must be one of "train", "val" or "test"')

        if isinstance(split, str):
            split_dir = self.splits[split]
            clz_files = list(map(lambda x: (x[0], list(map(lambda y: os.path.join(split_dir, y), x[1]))),
                                 self.lists[split].items()))
        elif isinstance(split, (list, tuple)):
            assert len(split) == 2
            clz_files = self._merge_clz_files(split[0], split[1].items())
        else:
            raise ValueError()

        if n_way is None and clz_to_sample is None:
            # sample all classes if not specified
            clz_to_sample = np.arange(len(clz_files))
        elif n_way is not None and clz_to_sample is None:
            clz_to_sample = np.arange(len(clz_files))
            np.random.shuffle(clz_to_sample)
            clz_to_sample = clz_to_sample[:n_way]

        support_set_x = []
        support_set_y = []
        query_set_x = []
        query_set_y = []
        idx_label = []
        n_sampled = n_support + n_query
        for i, j in enumerate(clz_to_sample):
            if len(clz_files[j][1]) >= n_sampled:
                img_sampled = np.random.choice(clz_files[j][1], size=n_sampled, replace=False)
            else:
                img_sampled = np.random.choice(clz_files[j][1], size=n_sampled, replace=True)

            s = list(map(lambda fn: self.loader(img_path=fn, img_gen=img_gen, target_size=target_size),
                         img_sampled))
            support_set_x.append(np.stack(s[:n_support]))
            support_set_y.append(i * np.ones((n_support, )))
            query_set_x.append(np.stack(s[-n_query:]))
            query_set_y.append(i * np.ones((n_query, )))
            idx_label.append(clz_files[j][0])

        # eventually get several 5D tensors [n_way, sample size for each class, width, height, channel]

        return {
            'support_x': support_set_x,
            'support_y': support_set_y,
            'query_x': query_set_x,
            'query_y': query_set_y,
            'idx_label': idx_label,
            'sample_clz': clz_to_sample
        }


def load_image_as_array(img_path, img_gen, target_size):
    """

    :param img_path: path to image file
    :param img_gen: (keras.preprocessing.image.ImageDataGenerator)
    :param target_size: (tuple) e.g. (299, 299)
    :return: 3D array of shape [width, height, n_channels]
    """
    img = load_img(img_path, grayscale=False, target_size=target_size)
    x = img_to_array(img, data_format=None)
    x = img_gen.random_transform(x)
    x = img_gen.standardize(x)
    return x


# noinspection PyUnusedLocal
def _load_bottleneck_features(img_path, **kwargs):
    """
    Load bottleneck features from text file

    :param img_path: path to text file
    :param kwargs:
    :return: 3D array of shape [width, height, n_channels]
    """
    with open(img_path + '.b', 'r') as f:
        line = f.read()

    return np.array([float(x) for x in line.split(',')])
