from common.util import get_bottleneck_config, get_dataset_info, ImageFlowGenerator, load_image_as_array
import json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from scripts.split_datasets import build_dataset_splits
import shutil
import tensorflow as tf
import tensorflow_hub as hub

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')


if __name__ == '__main__':
    images_dir = os.path.abspath(os.path.join(ROOT_DIR, 'data/devices'))
    splits_dir = os.path.abspath(os.path.join(ROOT_DIR, 'data/splits'))
    meta_filename = 'splits_info.json'
    bottleneck_dir = os.path.abspath(os.path.join(ROOT_DIR, 'data/bottleneck'))
    batch_size = 100
    tf_module = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    module_spec = hub.load_module_spec(tf_module)
    image_size, _ = hub.get_expected_image_size(module_spec)
    n_channels = hub.get_num_image_channels(module_spec)

    # split dataset and cache results on disk
    # noinspection PyBroadException
    try:
        print('Found splits, reading metadata')
        dataset_info = get_dataset_info(splits_dir)
    except Exception:
        print('Splits not found, building splits')
        dataset_info = build_dataset_splits(images_dir, splits_dir, external_dir=None,
                                            min_test=6, val_ratio=0.2, test_ratio=0.2, rebuild=False)

    meta_filename = os.path.join(bottleneck_dir, 'meta.json')
    if os.path.exists(bottleneck_dir):
        if os.path.exists(meta_filename):
            print('Reading bottleneck meta file')
            with open(meta_filename, 'r') as f:
                meta_info = json.load(f)
                is_consistent, external_consistency, meta_bottleneck = get_bottleneck_config(bottleneck_dir, splits_dir)
                if is_consistent:
                    print('Bottleneck split is consistent with original split, continue')
                    exit()
                else:
                    print('Consistency check failed, will delete current bottleneck directory')
                    shutil.rmtree(bottleneck_dir)
                    os.makedirs(bottleneck_dir)
        else:
            print('Meta file not found, rebuilding bottleneck dataset')
            shutil.rmtree(bottleneck_dir)
            os.makedirs(bottleneck_dir)
    elif os.path.exists(bottleneck_dir):
        print('History file not found, start fresh')
        os.makedirs(bottleneck_dir)

    # Prepare components for image generator, modify `ImageDataGenerator` to do data augmentation
    data_generator = ImageDataGenerator(rescale=1/255)
    image_flow_gen = ImageFlowGenerator(dataset_info)

    # Build graph
    x_in = tf.placeholder(shape=(None, image_size, image_size, n_channels), dtype=tf.float32)
    embedding_network = hub.Module(module_spec, trainable=False)
    embedding_network = embedding_network(x_in)
    init_op = tf.global_variables_initializer(), tf.local_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    for split in ['train', 'val', 'test', 'minor', 'rare']:
        split_lists = dataset_info['list']
        for clz, file_list in split_lists[split].items():
            for img in file_list:
                image_array = load_image_as_array(os.path.join(images_dir, img), data_generator,
                                                  target_size=(image_size, image_size))
                image_array = np.expand_dims(image_array, axis=0)
                bottleneck = sess.run(embedding_network, feed_dict={x_in: image_array})
                bottleneck = np.squeeze(bottleneck)
                bottleneck_clz_dir = os.path.join(bottleneck_dir, split, clz)
                if not os.path.exists(bottleneck_clz_dir):
                    os.makedirs(bottleneck_clz_dir)

                bottleneck_path = os.path.join(bottleneck_dir, split, img + '.b')
                bottleneck_str = ','.join(str(x) for x in bottleneck)
                with open(bottleneck_path, 'w') as f:
                    print('Cache bottleneck to', bottleneck_path)
                    f.write(bottleneck_str)

            print('Finish processing for', split, '/', clz)

        # Write meta info
        meta = {
            'bottleneck_split': {}
        }
        for it in os.listdir(splits_dir):
            tmp_path = os.path.join(splits_dir, it)
            if os.path.isfile(tmp_path):
                with open(tmp_path, 'r') as f:
                    d = json.load(f)

                if it == 'meta.json':
                    meta.update(d)
                    for k, v in d['split'].items():
                        ps = os.path.split(v)
                        meta['bottleneck_split'][k] = os.path.join(bottleneck_dir, ps[-1])

        with open(os.path.join(bottleneck_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=4)
