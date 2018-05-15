#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.


class Extractor:
    """
    Feature extractor class
    """

    def __init__(self, name, type):
        """
        Args:

        """
        self.name = name
        self.type = type

    def extract(self):
        """
        Should return a dictionary like:
        {'name': numpy.ndarray of shape [len(data),...]}
        """
        raise NotImplementedError
