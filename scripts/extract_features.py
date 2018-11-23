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


FEATURE = {
    'mfcc': extract_mfcc,
    'essentia': extract_essentia
}
EXTS = {
    extract_mfcc: '.npy',
    extract_essentia: '.json'
}
SAVE = {
    extract_mfcc: lambda x, fn: np.save(fn, x),
    extract_essentia: lambda x, fn: json.dump(x, open(fn, 'w'))
}


def _run_one(fn, out_path, feature):
    """ Helper function for extract one file"""
    global EXTS, SAVE
    try:
        out_fn = join(out_path, basename(fn).split('.')[0] + EXTS[feature])
        # extract feature
        m = feature(fn)
        SAVE[feature](m, out_fn)
    except Exception as e:
        print(e)


def run(fns, out_path, feature=extract_mfcc, n_workers=2):
    """ Extract MFCCs from given filenames

    Args:
        fns (list of str): list contains the all the filename of target files
    """
    assert feature in EXTS
    parmap(
        partial(_run_one, out_path=out_path, feature=feature),
        fns, n_workers=n_workers, verbose=True
    )


if __name__ == "__main__":

    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("in_root", help='root where all the audio files located')
    parser.add_argument("out_root", help='root for output files')
    parser.add_argument("--feature", type=str, default='mfcc',
                        help="type of feature {'mfcc', 'essentia'}")
    parser.add_argument("--n-workers", type=int, default=2,
                        help="number of workers")
    args = parser.parse_args() 

    # prepare output directory if not there
    pathlib.Path(dirname(args.out_root)).mkdir(parents=True, exist_ok=True) 

    # get the filenames
    fns = glob.glob(join(args.in_root, '*/*.mp3'))

    # process
    run(fns, args.out_root, FEATURE[args.feature], args.n_workers)
