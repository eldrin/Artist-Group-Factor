import os, sys, glob, argparse, pathlib, json
from os.path import join, dirname, basename
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from functools import partial

import numpy as np
from tqdm import tqdm

from agf.feature import extract_mfcc, extract_essentia
from agf.config import Config as cfg
from agf.utils import parmap


def agf(feature_path, type='mfcc'):
    """"""
    # load the 

