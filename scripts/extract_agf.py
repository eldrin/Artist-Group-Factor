import os, sys, glob, argparse, pathlib, json
from os.path import join, dirname, basename
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle as pkl

from agf.process import MFCCAGF, DMFCCAGF, EssentiaAGF, SubGenreAGF

AGF_TYPES = {
    'mfcc': MFCCAGF,
    'dmfcc': DMFCCAGF,
    'essentia': EssentiaAGF,
    'subgenre': SubGenreAGF
}

EXT_TYPE = {
    'mfcc': '*.npy',
    'dmfcc': '*.npy',
    'essentia': '*.json',
}


if __name__ == "__main__":

    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("agf_type", type=str, default='mfcc',
                        choices=set(AGF_TYPES.keys()),
                        help='type indication for AGF \
                            {mfcc, dmfcc, essentia, subgenre}')
    parser.add_argument("feature_root",
                        help='path for feature files corresponding to AGF type')
    parser.add_argument("metadata", help='path to "track.csv"')
    parser.add_argument("out_fn", help='filename for output files')
    args = parser.parse_args() 

    # prepare output directory if not there
    pathlib.Path(dirname(args.out_fn)).mkdir(parents=True, exist_ok=True) 

    # process
    proc = AGF_TYPES[args.agf_type](args.metadata, verbose=True)
    if isinstance(proc, SubGenreAGF):
        result = proc.process()
    else:
        fns = glob.glob(join(args.feature_root, EXT_TYPE[args.agf_type]))
        result = proc.process(fns)

    # save the output
    with open(args.out_fn, 'wb') as f:
        pkl.dump(result, f)
