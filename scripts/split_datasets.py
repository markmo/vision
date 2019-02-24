from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import json
import numpy as np
import os
import shutil

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')


def _prepare_or_get_split_lists(images_dir, external_dir=None, min_test=6, val_ratio=0.2, test_ratio=0.2):
    """
    This method will read the `images_dir`, which is organised as follows:
    ::

        <images_dir>/
            <class_1>/
                001.jpg
                002.jpg
                ...
            <class_n>/
                001.jpg
                002.jpg
                ...

    :param images_dir:
    :param external_dir:
    :param min_test: classes with number examples < `min_test` will be cut
    :param val_ratio:
    :param test_ratio:
    :return: (dict) of filenames for 'train', 'val', and 'test'
    """
    images_path = os.path.abspath(os.path.join(ROOT_DIR, images_dir))
    if not os.path.exists(images_path):
        raise Exception('{} not found'.format(images_path))

    datasets = {}
    dir_tree = list(os.walk(images_path))
    labels, sample_counts = [], []
    train = {}
    val = {}
    test = {}
    minor = {}
    rare = {}
    for i, content in enumerate(dir_tree[1:]):
        prefix = content[0]
        label = prefix.split('/')[-1]
        assert label == dir_tree[0][1][i]
        labels.append(label)
        n_samples = len(content[2])
        sample_counts.append(n_samples)

        # filter out classes with too few examples (n_samples < min_test)
        if n_samples >= (min_test / test_ratio) or n_samples >= 3 * min_test:
            datasets[label] = list(map(lambda f: os.path.join(label, f), content[2]))
        elif n_samples >= min_test:
            # minority class all goes to test set
            minor[label] = list(map(lambda f: os.path.join(label, f), content[2]))
        else:
            rare[label] = list(map(lambda f: os.path.join(label, f), content[2]))

    for label, flist in datasets.items():
        np.random.shuffle(flist)
        train[label] = []
        val[label] = []
        test[label] = []
        for img in flist:
            len_val = len(val[label])
            cut_val = val_ratio * len(flist)
            len_test = len(test[label])
            cut_test = (test_ratio + val_ratio) * len(flist)
            if len_val < cut_val:
                val[label].append(img)
            elif len_test < cut_test:
                test[label].append(img)
            else:
                train[label].append(img)

    external = None
    if external_dir:
        external_path = os.path.abspath(os.path.join(ROOT_DIR, external_dir))
        flist = list(os.walk(external_path))[1:]
        external = list(map(lambda x: (x[0].split('/')[-1],
                                       list(map(lambda f: os.path.join(x[0].split('/')[-1], f), x[2]))), flist))
        external = dict(external)

    return {
        'train': train,
        'val': val,
        'test': test,
        'minor': minor,
        'rare': rare
    }, external


def build_dataset_splits(images_dir, splits_dir, external_dir=None,
                         min_test=6, val_ratio=0.2, test_ratio=0.2, rebuild=False, seed=None):
    meta_path = os.path.join(ROOT_DIR, splits_dir, 'meta.json')
    info = None
    splits_path = os.path.abspath(os.path.join(ROOT_DIR, splits_dir))
    images_path = os.path.abspath(os.path.join(ROOT_DIR, images_dir))
    if rebuild and os.path.exists(splits_path):
        print('Rebuilding; removing existing files')
        shutil.rmtree(splits_path)
        os.makedirs(splits_path)
    elif not rebuild and os.path.exists(splits_path):
        if os.path.exists(meta_path):
            print('Read meta file')
            with open(meta_path, 'r') as f:
                info = json.load(f)
                settings = info['settings']
                if (settings['val_ratio'] == val_ratio
                        and settings['test_ratio'] == test_ratio
                        and settings['seed'] == seed):
                    assert 'list' in info.keys()
                    assert 'split' in info.keys()
                    print('Parameters consistent; continue')
                    return info
                else:
                    print('Parameters have changed; rebuild')
                    shutil.rmtree(splits_path)
                    os.makedirs(splits_path)
                    info = None
        else:
            print('Meta file not found; rebuild')
            shutil.rmtree(splits_path)
            os.makedirs(splits_path)
    elif not os.path.exists(splits_path):
        os.makedirs(splits_path)

    if not info:
        info = {'settings': {
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': seed
        }}
        np.random.seed(seed)
        info['list'], external_list = \
            _prepare_or_get_split_lists(images_dir, external_dir, min_test, val_ratio, test_ratio)
        np.random.seed()  # reset random state
        print('Copying files')
        split_dirs = {}
        for split, clzlist in info['list'].items():
            split_dirs[split] = os.path.join(splits_path, split)
            for clz, flist in clzlist.items():
                os.makedirs(os.path.join(splits_path, split, clz))
                for f in flist:
                    new_path = os.path.join(splits_path, split, f)
                    old_path = os.path.join(images_path, f)
                    shutil.copy(old_path, new_path)

        if external_list:
            split_dirs['external'] = external_dir
            info['list']['external'] = external_list

        info.update({'split': split_dirs})
        with open(meta_path, 'w') as f:
            json.dump(info, f, indent=4)
    else:
        with open(meta_path, 'r') as f:
            print('Loading existing meta info')
            info = json.load(f)

    return info


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), '../src', 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    build_dataset_splits(
        images_dir=constants['images_dir'],
        splits_dir=constants['splits_dir'],
        external_dir=constants['images_dir'],
        min_test=constants['min_test'],
        val_ratio=constants['val_ratio'],
        test_ratio=constants['test_ratio'],
        rebuild=constants['rebuild']
    )


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Split datasets')
    parser.add_argument('--val-ratio', dest='val_ratio', type=float, help='validation split ratio')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float, help='test split ratio')
    parser.add_argument('--seed', dest='seed', type=int, help='seed for randomization')
    parser.add_argument('--rebuild', dest='rebuild', help='indicator to rebuild datasets', action='store_true')
    parser.set_defaults(rebuild=False)
    args = parser.parse_args()

    run(vars(args))
