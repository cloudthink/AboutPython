#-*- coding: UTF-8 -*-
import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from DistBase import DistMixin, DistMeta
from AutoNN.DistNN import AutoBasic, AutoAdvanced


class DistBasic(AutoBasic, DistMixin, metaclass=DistMeta):
    pass


class DistAdvanced(AutoAdvanced, DistMixin, metaclass=DistMeta):
    pass
