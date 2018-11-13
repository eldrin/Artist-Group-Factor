from os.path import basename, join, dirname

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
from essentia.standard import *

from tqdm import tqdm

from .utils import MetaDataLoader
from .config import Config as cfg


AGF_TYPES = {'m': 'mfcc',
             'd': 'dmfcc',
             'e': 'essentia',
             's': 'subgenre'}


class AGF:
    """ Artist Group Factor Extractor """
    def __init__(self, metadata_fn, sample_rate=cfg.SR, num_clusters=cfg.K,
                 num_factors=cfg.R, num_mels=cfg.M, num_mfccs=cfg.D,
                 num_chromas=cfg.C, verbose=False):
        # load variables
        self.sample_rate = sample_rate
        self.num_clusters = num_clusters
        self.num_factors = num_factors
        self.num_mels = num_mels
        self.num_mfccs = num_mfccs
        self.num_chromas = num_chromas
        self.verbose = verbose

        # load metadata
        self.metadata = MetaDataLoader(metadata_fn)

    def extract(self, audio_fns, agf_type='m'):
        """
        Args:
            audio_fn (list of str): list of filenames of audio
            artist_audio_map (dict):
                dictionary containing maps from artist (int. index) to
                relevant songs (list of int. indices)
            type (str): flag for the AGF type {'m', 'd', 'e', 's'}
        """
        self._verify_agf_type(agf_type)

        # convert list of fns into dictionary of name/fns
        audio_fns = {int(basename(fn).split('.mp3')[0]):fn for fn in audio_fns}

        if agf_type == 'm':
            return extract_mfcc_agf(audio_fns, self.metadata.artist_audio_map,
                                    sr=self.sample_rate,
                                    m=self.num_mels, d=self.num_mfccs,
                                    k=self.num_clusters, r=self.num_factors,
                                    verbose=self.verbose)
        elif agf_type == 'd':
            raise NotImplementedError()

        elif agf_type == 'e':
            raise NotImplementedError()

        elif agf_type == 's':
            raise NotImplementedError()

        else:
            raise NotImplementedError()
            
    def _verify_agf_type(self, type):
        """"""
        assert type in AGF_TYPES


def extract_mfcc_agf(audio_fns, artist_audio_map, sr, m, d, k, r, verbose=False):
    """ Extract MFCC based AGF

    Args:
        audio_fns (list of str): list containing all target audio files
        artist_audio_map (dict):
            dictionary containing maps from artist (int. index) to
            relevant songs (list of int. indices)
        sr (int): sampling rate
        m (int): number of mel bands
        d (int): number of mfcc coefficients
        k (int): number of clusters for K-Means
        r (int): number of factors for LDA
        verbose (bool): verbosity
    """
    # 1. get universal feature model
    kms = MiniBatchKMeans(n_clusters=k)
    mfccs = {}

    iterator = artist_audio_map.items()
    if verbose:
        print('Getting global Kmeans model...')
        iterator = tqdm(iterator, ncols=80)

    # process!
    for artist_id, track_ids in iterator:
        # get MFCCs per artist
        mfccs[artist_id] = []
        for track_id in track_ids:
            fn = audio_fns[track_id]
            # add every frames from all songs
            for frame in FrameGenerator(MonoLoader(filename=fn, sampleRate=sr)(),
                                        frameSize=cfg.WINSZ, hopSize=cfg.HOPSZ):
                mfccs[artist_id].append(
                    MFCC(numberBands=m, numberCoefficients=d, sampleRate=sr)(
                        Spectrum(size=len(frame))(frame)
                    )[1]  # only takes the MFCCs
                )
        mfccs[artist_id] = np.array(mfccs[artist_id])

        # update K-Means
        kms.partial_fit(mfccs[artist_id])

    # get individual songs Bag-of-Features
    iterator = artist_audio_map.items()
    if verbose:
        print('Getting global Kmeans model...')
        iterator = tqdm(iterator, ncols=80)

    # define fn-integer map for further process
    artist2ind = dict([(v, k) for k, v in enumerate(artist_audio_map.keys())])
    raw_sparse_v = []
    raw_sparse_i = []
    raw_sparse_j = []
    for artist_id, track_id in iterator:
        # for the normalization over the # of songs per artist
        n_frames = mfccs[artist_id].shape[0]
        for k, freq in sorted(Counter(kms.predict(mfccs[artist_id])).items(),
                              key=lambda k: k[0]):
            # register data
            raw_sparse_v.append(artist2ind[artist])
            raw_sparse_i.append(k)
            raw_sparse_j.append(freq/n_frames)

    # build sparse data matrix (n_audio, n_clusters)
    Xs = sp.coo_matrix((raw_sparse_v, (raw_sparse_i, raw_sparse_j)),
                       shape=(len(artist2ind), k)).tocsr()

    # initiate LDA & get the AGF
    lda = LatentDirichletAllocation(n_components=r, n_jobs=cfg.NJOBS)
    zs = lda.fit_transform(Xs)

    return zs, kms, lda
