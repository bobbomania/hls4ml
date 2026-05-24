
from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import BayesianDropout
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Bayesian Dropout template

dropout_config_template = """
struct config{index} : nnet::dropout_config {{
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static constexpr float drop_rate = {drop_rate};
    std::default_random_engine eng = std::default_random_engine();
}};\n"""

dropout_function_template = 'nnet::dropout<{input_t}, {output_t}, {config}>({input}, {output});'

dropout_include_list = ['nnet_utils/nnet_dropout.h', 'nnet_utils/nnet_dropout_stream.h']

class DropoutConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(BayesianDropout)
        self.template = dropout_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)

class DropoutFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(BayesianDropout, include_header=dropout_include_list)
        self.template = dropout_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)
