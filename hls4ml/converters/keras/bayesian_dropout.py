from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler


@keras_handler('BayesianDropout')
def parse_bayesian_dropout_layer(keras_layer, input_names, input_shapes, data_reader):
    assert('BayesianDropout' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    layer['drop_rate'] = keras_layer['config']['drop_rate']
    layer['seed'] = keras_layer['config']['seed']

    return layer, [shape for shape in input_shapes[0]]
