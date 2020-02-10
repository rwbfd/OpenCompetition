# !/user/bin/python
# -*- coding:utf-8 -*-

'''
https://github.com/allenai/allennlp/blob/master/allennlp/modules/augmented_lstm.py
modify by LAQ
'''

import torch
from file_utils import TaskModelBase


class HLstm(TaskModelBase):
    def __init__(self, input_size, hidden_size, use_input_projection_bias=True):
        super(HLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_input_projection_bias = use_input_projection_bias

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=self.use_input_projection_bias)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)

    def forward(self, inputs):

        initial_state = None

        # inputs = self.embedding(inputs)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        output_accumulator = inputs.new_zeros(batch_size, seq_len, self.hidden_size)
        if initial_state is None:
            previous_memory = inputs.new_zeros(batch_size, self.hidden_size)
            previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        # current_length_index = batch_size - 1 if self.go_forward else 0
        # if self.recurrent_dropout_probability > 0.0:
        #     dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_memory)
        # else:
        #     dropout_mask = None

        for timestep in range(seq_len):
            # The index depends on which end we start.
            index = timestep
            timestep_input = inputs[:, index]

            # Do the projections for all the gates all at once.
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                       projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                        projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                     projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                        projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * torch.tanh(memory)

            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            previous_memory = memory
            previous_state = timestep_output
            output_accumulator[:, index] = timestep_output



        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (previous_state.unsqueeze(0),
                       previous_memory.unsqueeze(0))

        return output_accumulator
