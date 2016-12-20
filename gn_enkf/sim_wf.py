#!/usr/bin/env python

import math
import time
import numpy as np
import sys
import pickle

from wf_state import State

spath = sys.argv[1]

with open(spath, 'rb') as f:
	up = pickle.Unpickler(f)
	s = up.load() #stations

# run simulation
s.fx()

with open(spath, 'wb') as f:
	p = pickle.Pickler(f,protocol=pickle.HIGHEST_PROTOCOL)
	p.dump(s)
