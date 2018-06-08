#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
"""

class Features:
    """
    Class used to extract and manage features
    """

    def __init__(self):
        self.extractors = []
        self.kwargs = []

    def add_extractors(self, extractor_list):
        for e in extractor_list:
            self.extractors.append(e['name'])
            self.kwargs.append(e['kwargs'])
            