#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.


class Extractor:
    """
    Feature extractor class
    """

    def __init__(self, name):
        self.name = name
        print("Create {0}".format(self.name))

    def extract(self, data_raw, chars, words, **kwargs):
        """
        Should return a dictionary like:
        {'name': numpy.ndarray of shape [len(data),...]}
        """
        raise NotImplementedError
