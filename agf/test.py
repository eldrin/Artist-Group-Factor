from .data import MetaDataLoader
from .process import *
import glob


METADATA_FN = '/home/ubuntu/bulk/dataset/FMA/fma_metadata/tracks.csv'


def test_agf_mfcc():
    """"""
    fns = glob.glob('/home/ubuntu/bulk/dataset/FMA/fma_mfcc/*.npy')
    proc = MFCCAGF(METADATA_FN, verbose=True)
    result = proc.process(fns)
    return result

def test_agf_dmfcc():
    """"""
    fns = glob.glob('/home/ubuntu/bulk/dataset/FMA/fma_mfcc/*.npy')
    proc = DMFCCAGF(METADATA_FN, verbose=True)
    result = proc.process(fns)
    return result

def test_agf_ess():
    """"""
    fns = glob.glob('/home/ubuntu/bulk/dataset/FMA/fma_essentia/*.json')
    proc = EssentiaAGF(METADATA_FN, verbose=True)
    result = proc.process(fns)
    return result

def test_agf_subgenre():
    """"""
    proc = SubGenreAGF(METADATA_FN, verbose=True)
    result = proc.process()
    return result


