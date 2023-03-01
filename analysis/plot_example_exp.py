import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import analysis.util as util
import analysis.v2i_bw as v2i_bw, analysis.trajectory as trajectory, analysis.disconnection as disconnection

import run_experiment

font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)