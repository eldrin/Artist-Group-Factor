from multiprocessing import Pool

import essentia
essentia.log.infoActive = False
from essentia.standard import *

import librosa
import numpy as np

from tqdm import tqdm

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_MAP = {key: one_hot for key, one_hot in zip(KEYS, np.eye(len(KEYS)))}
MODES = ['major', 'minor']
MODE_MAP = {mode: one_hot for mode, one_hot in zip(MODES, np.eye(len(MODES)))}


def audioread(fn, sr):
    """"""
    return librosa.load(fn, sr=sr)[0]


def pool2dict(pool, flatten=True):
    """ Convert Pool object to dictionary

    Args:
        pool (essentia.Pool): contains all the feature output from extractors
        flatten (bool): flag for flattening the dict value when the entry is vector
    """
    output = {}
    for key in pool.descriptorNames():
        if key == 'rhythm.beats_position':
            continue
        if key.split('.')[0] == 'metadata':
            continue
        if 'cov' in key.split('.')[-1]:
            # U, S, V = np.linalg.svd(a[key])
            # maybe doing summarization by PCA
            continue
        else:
            if isinstance(pool[key], np.ndarray) and pool[key].shape[0] > 1:
                for i, val in enumerate(pool[key]):
                    output[key + '_{:d}'.format(i)] = val
            elif pool[key] in KEYS:
                for k, val in enumerate(KEY_MAP[pool[key]]):
                    output[key + '_{}'.format(KEYS[k])] = val
            elif pool[key] in MODES:
                for k, val in enumerate(MODE_MAP[pool[key]]):
                    output[key + '_{}'.format(MODES[k])] = val
            else:
                output[key] = pool[key]

    # force converting
    for k, v in output.items():
        output[k] = float(v)

    return output


def parmap(func, iterable, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """
    
    if n_workers == 1:
        if verbose:
            iterable = tqdm(iterable, total=len(iterable), ncols=80)
        return map(func, iterable)
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm(total=len(iterable), ncols=80) as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)
