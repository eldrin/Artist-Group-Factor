import librosa
from essentia.standard import *

from .config import Config as cfg
from .utils import audioread, pool2dict


def extract_mfcc(fn, sr=cfg.SR, n_fft=cfg.WINSZ, hop_sz=cfg.HOPSZ,
                 n_mels=cfg.M, n_mfccs=cfg.D, delta=False):
    """ Extract MFCC from one filename

    Args:
        fn (str): path to the audio file
        sr (float): sampling rate
        n_fft (int): window size for STFT
        hop_sz (int): hop size for STFT
        n_mels (int): number of bands for mel scaling
        n_mfccs (int): number of coefficients for MFCC
        delta (bool): flag indicates the function returns time delta
    """
    y = audioread(fn, sr)
    m = _mfcc(y)
    if delta:
        return m[1:] - m[:-1]
    else:
        return m


def _mfcc(audio, n_mfccs=13, n_bands=40, sr=22050, winsz=2048, hopsz=1024):
    """"""
    return librosa.feature.mfcc(audio, sr=sr, n_fft=winsz, hop_length=hopsz,
                                n_mfcc=n_mfccs, n_mels=n_bands).T


def extract_essentia(fn, sr=44100):
    """ Extract Essentia's Music Extractor feature set (v2)

    Args:
        fn (str): filename
    """
    return pool2dict(MusicExtractor(analysisSampleRate=sr)(fn)[0])
