from argparse import ArgumentParser
from common.util import get_bottleneck_config, ImageFlowGenerator, load_hyperparams, merge_dict
from common.util import run_training
from keras.preprocessing.image import ImageDataGenerator
from model_setup import SemiHardModel
import os
import tensorflow_hub as hub
from time import time


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    data_generator = ImageDataGenerator(rescale=1/255, rotation_range=90, width_shift_range=0.2,
                                        height_shift_range=0.2, horizontal_flip=True)
    print('Load', constants['module_spec'])
    module_spec = hub.load_module_spec(constants['module_spec'])
    image_size, _ = hub.get_expected_image_size(module_spec)
    # n_channels = hub.get_num_image_channels(module_spec)
    # project_dir = 'tmp/semihard_full_' + 'time:' + str(int(time()))[-3:] +\
    #               '/top:lambda:' + str(constants['lambda_reg']) +\
    #               'margin:' + str(constants['tl_margin'])
    project_dir = '/Users/d777710/src/DeepLearning/vision'
    print('Project dir:', project_dir)
    _, _, bottleneck_config = get_bottleneck_config(os.path.join(project_dir, constants['bottleneck_dir']),
                                                    os.path.join(project_dir, constants['splits_dir']))
    bottleneck_flow_gen = ImageFlowGenerator(bottleneck_config, mode='bottleneck')
    constants.update({
        'train_dir': os.path.join(project_dir, constants['train_subdir']),
        'top_model_dir': os.path.join(project_dir, constants['top_model_subdir']),
        'val_dir': os.path.join(project_dir, constants['val_subdir']),
        'top_model_val_dir': os.path.join(project_dir, constants['top_model_val_subdir']),
        'data_flow_gen': bottleneck_flow_gen,
        'eval_every_n_steps': 5,
        'generator': data_generator,
        'image_size': image_size
    })
    model = SemiHardModel(constants, train_top_only=True)
    run_training(model, constants)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Vision model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--train', dest='train', help='training mode', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
