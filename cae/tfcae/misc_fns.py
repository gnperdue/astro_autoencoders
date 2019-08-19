import logging
import json
import inspect

import tensorflow as tf

LOGGER = logging.getLogger(__name__)


def get_expected_image_shape(filename):
    shapes_dict = {
        'stargalaxy_real': (48, 48, 3),
        'stargalaxy_sim_20190214': (64, 64, 1),
        'strong_lensing_spacebased': (101, 101, 1)
    }
    for k, v in shapes_dict.items():
        if k in filename:
            return v
    return None


def get_logging_level(log_level):
    log_level = log_level.upper()
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')

    return logging_level


def get_number_of_trainable_parameters():
    """ use default graph """
    # https://stackoverflow.com/questions/38160940/ ...
    LOGGER.debug('Now compute total number of trainable params...')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        LOGGER.debug(' layer name = {}, shape = {}, n_params = {}'.format(
            name, shape, variable_parameters
        ))
        total_parameters += variable_parameters
    LOGGER.debug('Total parameters = %d' % total_parameters)
    return total_parameters


def log_function_args(vs, logger):
    logger.info('Calling {}'.format(inspect.stack()[1][3]))
    logger.info(json.dumps(
        vs, indent=3, skipkeys=True, default=repr, sort_keys=True
    ))
