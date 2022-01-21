# -*- coding: utf-8 -*-

import sys

# Force the "slow" version for cloudpickle_generators
from cloudpickle import cloudpickle
sys.modules['cloudpickle'] = cloudpickle

from cloudpickle import *
from cloudpickle import CloudPickler as Pickler

import cloudpickle_generators
cloudpickle_generators.register()
