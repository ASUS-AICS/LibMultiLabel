import h5py
import torch

def set_value(model):
    with h5py.File('../lmtc-emnlp2020/data/models/EURLEX57K_ZERO-BIGRU-LWAN.h5', "r") as fp:
        state_dict = {
            # 'network.embedding.embedding.weight': 0,
            'network.encoder.rnn.weight_ih_l0': fp['model_weights']['bidirectional']['bidirectional']['forward_gru']['gru_cell_1']['kernel:0'].value.T,
            'network.encoder.rnn.weight_hh_l0': fp['model_weights']['bidirectional']['bidirectional']['forward_gru']['gru_cell_1']['recurrent_kernel:0'].value.T,
            'network.encoder.rnn.bias_ih_l0': fp['model_weights']['bidirectional']['bidirectional']['forward_gru']['gru_cell_1']['bias:0'][0],
            'network.encoder.rnn.bias_hh_l0': fp['model_weights']['bidirectional']['bidirectional']['forward_gru']['gru_cell_1']['bias:0'][1],
            'network.encoder.rnn.weight_ih_l0_reverse': fp['model_weights']['bidirectional']['bidirectional']['backward_gru']['gru_cell_2']['kernel:0'].value.T,
            'network.encoder.rnn.weight_hh_l0_reverse': fp['model_weights']['bidirectional']['bidirectional']['backward_gru']['gru_cell_2']['recurrent_kernel:0'].value.T,
            'network.encoder.rnn.bias_ih_l0_reverse': fp['model_weights']['bidirectional']['bidirectional']['backward_gru']['gru_cell_2']['bias:0'][0],
            'network.encoder.rnn.bias_hh_l0_reverse': fp['model_weights']['bidirectional']['bidirectional']['backward_gru']['gru_cell_2']['bias:0'][1],
            'network.attention.linear.weight': fp['model_weights']['zero_label_wise_attention']['zero_label_wise_attention']['zero_label_wise_attention_Wd:0'].value.T,
            'network.attention.linear.bias': fp['model_weights']['zero_label_wise_attention']['zero_label_wise_attention']['zero_label_wise_attention_bd:0'].value,
            'network.output.linear.weight': fp['model_weights']['dense']['dense']['kernel:0'].value.T,
            'network.output.linear.bias': fp['model_weights']['dense']['dense']['bias:0'].value
        }
    for k, v in state_dict.items():
        state_dict[k] = torch.Tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model
