__author__ = 'max'
__maintainer__ = 'takashi'

from typing import Tuple, Callable, Union, List
import torch
from torch import Tensor
import torch.nn as nn


def var_lstm_cell(input: Tensor, hidden: Tuple[Tensor, Tensor], w_ih: Tensor, w_hh: Tensor,
                  b_ih: Tensor = None, b_hh: Tensor = None, noise_in: Tensor = None, noise_hidden: Tensor = None) \
        -> Tuple[Tensor, Tensor]:
    input = input.expand(4, *input.size()) if noise_in is None else input.unsqueeze(0) * noise_in

    hx, cx = hidden
    hx = hx.expand(4, *hx.size()) if noise_hidden is None else hx.unsqueeze(0) * noise_hidden

    gates = torch.add(torch.baddbmm(b_ih.unsqueeze(1), input, w_ih), torch.baddbmm(b_hh.unsqueeze(1), hx, w_hh))

    ingate, forgetgate, cellgate, outgate = gates

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = torch.add(torch.mul(forgetgate, cx), torch.mul(ingate, cellgate))
    hy = torch.mul(outgate, torch.tanh(cy))

    return hy, cy


def var_masked_recurrent(reverse: bool = False) \
        -> Callable[[Tensor, Union[Tensor, Tuple[Tensor, Tensor]], nn.Module, Tensor], Tuple[Tensor, Tensor]]:

    def forward(input: Tensor, hidden: Union[Tensor, Tuple[Tensor, Tensor]], cell: nn.Module, mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = cell(input[i], hidden)
            if mask is not None and mask[i].data.min() == 0:
                # hack to handle LSTM
                if isinstance(hidden, tuple):
                    hx, cx = hidden
                    float_mask = (mask[i] != 0).float()
                    hidden = (hx * float_mask, cx * float_mask)
                else:
                    hidden = hidden * (mask[i] != 0).float()
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(tuple(output), 0).view((input.size(0), *output[0].size()))

        return hidden, output

    return forward


def stacked_rnn(inners: Tuple, num_layers: int, lstm: bool = False) \
        -> Callable[[Tensor, Union[Tensor, Tuple[Tensor, Tensor]], List[nn.Module], Tensor], Tuple[Tensor, Tensor]]:
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input: Tensor, hidden: Union[Tensor, Tuple[Tensor, Tensor]], cells: List[nn.Module], mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        assert (len(cells) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input, hidden[l], cells[l], mask)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(tuple(all_output), input.dim() - 1)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view((total_layers, *next_h[0].size())),
                torch.cat(next_c, 0).view((total_layers, *next_c[0].size()))
            )
        else:
            next_hidden = torch.cat(tuple(next_hidden), 0).view((total_layers, *next_hidden[0].size()))

        return next_hidden, input

    return forward


def autograd_var_masked_rnn(num_layers: int = 1, batch_first: bool = False,
                            bidirectional: bool = False, lstm: bool = False) \
        -> Callable[[Tensor, List[nn.Module], Union[Tensor, Tuple[Tensor, Tensor]], Tensor],
                    Tuple[Tensor, Tensor]]:
    rec_factory = var_masked_recurrent

    if bidirectional:
        layer = (rec_factory(), rec_factory(reverse=True))
    else:
        layer = (rec_factory(),)

    func = stacked_rnn(layer, num_layers, lstm=lstm)

    def forward(input: Tensor, cells: List[nn.Module], hidden: Union[Tensor, Tuple[Tensor, Tensor]], mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        if batch_first:
            input = input.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        nexth, output = func(input, hidden, cells, mask)

        if batch_first:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


def var_masked_step() -> Callable[[Tensor, Union[Tensor, Tuple[Tensor, Tensor]], nn.Module, Tensor],
                                  Tuple[Tensor, Tensor]]:
    def forward(input: Tensor, hidden: Union[Tensor, Tuple[Tensor, Tensor]], cell: nn.Module, mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        hidden = cell(input, hidden)
        if mask is not None and mask.data.min() == 0:
            # hack to handle LSTM
            if isinstance(hidden, tuple):
                hx, cx = hidden
                float_mask = (mask != 0).float()
                hidden = (hx * float_mask, cx * float_mask)
            else:
                hidden = hidden * (mask != 0).float()
        # hack to handle LSTM
        output = hidden[0] if isinstance(hidden, tuple) else hidden

        return hidden, output

    return forward


def stacked_step(layer: Callable[[Tensor, Union[Tensor, Tuple[Tensor, Tensor]], nn.Module, Tensor],
                                 Tuple[Tensor, Tensor]],
                 num_layers: int, lstm: bool = False) \
        -> Callable[[Tensor, Union[Tensor, Tuple[Tensor, Tensor]], List[nn.Module], Tensor], Tuple[Tensor, Tensor]]:

    def forward(input: Tensor, hidden: Union[Tensor, Tuple[Tensor, Tensor]], cells: List[nn.Module], mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        assert (len(cells) == num_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for l in range(num_layers):
            hy, output = layer(input, hidden[l], cells[l], mask)
            next_hidden.append(hy)
            input = output

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view((num_layers, *next_h[0].size())),
                torch.cat(next_c, 0).view((num_layers, *next_c[0].size()))
            )
        else:
            next_hidden = torch.cat(tuple(next_hidden), 0).view((num_layers, *next_hidden[0].size()))

        return next_hidden, input

    return forward


def autograd_var_masked_step(num_layers: int = 1, lstm: bool = False) \
        -> Callable[[Tensor, List[nn.Module], Union[Tensor, Tuple[Tensor, Tensor]], Tensor], Tuple[Tensor, Tensor]]:
    layer = var_masked_step()

    func = stacked_step(layer, num_layers, lstm=lstm)

    def forward(input: Tensor, cells: List[nn.Module], hidden: Union[Tensor, Tuple[Tensor, Tensor]], mask: Tensor) \
            -> Tuple[Tensor, Tensor]:
        nexth, output = func(input, hidden, cells, mask)
        return output, nexth

    return forward
