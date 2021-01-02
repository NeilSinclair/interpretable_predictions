'''
To do: utils with shift_tokens_right()
       include a <CLS> token in the tokens

'''

import torch
from torch import nn
from torch.nn import LSTMCell
import numpy as np
from latent_rationale.nn.kuma_gate import KumaGate
from latent_rationale.common.util import get_encoder
from latent_rationale.nn.rcnn import RCNNCell


class IndependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 768,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 distribution: str = "kuma"
                 ):
        # pass the instantiated BartModel to the function
        super(IndependentLatentModel, self).__init__()

        self.layer = layer

        enc_size = hidden_size

        if distribution == "kuma":
            self.z_layer = KumaGate(enc_size)
        else:
            raise ValueError("unknown distribution")

        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, h, mask, **kwargs):

        # encode sentence
        lengths = mask.sum(1)
        z_dist = self.z_layer(h)

        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z


class DependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self,
                 embed:       nn.Embedding = None,
                 hidden_size: int = 768,
                 dropout:     float = 0.1,
                 layer:       str = "rcnn",
                 z_rnn_size:  int = 30
                 ):

        super(DependentLatentModel, self).__init__()

        self.layer = layer

        enc_size = hidden_size

        if layer == "rcnn":
            self.z_cell = RCNNCell(enc_size + 1, z_rnn_size)
        else:
            self.z_cell = LSTMCell(enc_size + 1, z_rnn_size)

        self.z_layer = KumaGate(enc_size + z_rnn_size)

        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, h, mask):

        # encode sentence
        batch_size, time = h.size()
        lengths = mask.sum(1)

        h = h.transpose(0, 1)  # time, batch, dim

        z = []
        z_dists = []

        # initial states  [1, B, z_rnn_dim]
        if isinstance(self.z_cell, LSTMCell):  # LSTM
            state = h.new_zeros(
                [2 * batch_size, self.z_cell.hidden_size]).chunk(2)
        else:  # RCNN
            state = h.new_zeros([3 * batch_size, self.z_cell.hidden_size]).chunk(3)

        for h_t, t in zip(h, range(time)):
            ### NOTE: THIS MIGHT NOT WORK AS PLANNED BECAUSE OF THE CROSS/SELF-ATTENTION OF
            ### THE FINAL HIDDEN STATE, BUT LET'S SEE
            # compute Binomial z distribution for this time step
            z_t_dist = self.z_layer(torch.cat([h_t, state[0]], dim=-1))
            z_dists.append(z_t_dist)

            # we sample once since the state was already repeated num_samples
            if self.training:
                z_t = z_t_dist.sample()  # [B, 1]
            else:
                # deterministic strategy
                p0 = z_t_dist.pdf(h.new_zeros(()))
                p1 = z_t_dist.pdf(h.new_ones(()))
                pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
                zero_one = torch.where(
                    p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
                z_t = torch.where((pc > p0) & (pc > p1),
                                  z_t_dist.mean(), zero_one)  # [B, M]

            z.append(z_t)

            # update cell state (to make dependent decisions)
            rnn_input = torch.cat([h_t, z_t], dim=-1)  # [B, 2D+1]
            state = self.z_cell(rnn_input, state)

        z = torch.stack(z, dim=1).squeeze(-1)  # [B, T]
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z
        self.z_dists = z_dists

        return z


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        Taken directly from modeling_bart.py from HF repository """

    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]

    return prev_output_tokens