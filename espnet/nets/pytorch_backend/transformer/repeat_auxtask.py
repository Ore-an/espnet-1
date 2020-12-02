#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import torch


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args, last_layer=None):
        """Repeat."""
        assert last_layer is None or last_layer <= len(self)
        lcount = 0
        for m in self:
            args = m(*args)
            lcount += 1
            if lcount == last_layer:
                return args
        return args


def repeat(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(N)])
