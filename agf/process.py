import json
import os
from os.path import basename, join, dirname
import glob
from collections import Counter
from functools import partial
from itertools import chain
import pickle as pkl

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans, KMeans
import pandas as pd

from tqdm import tqdm
tqdm80 = partial(tqdm, ncols=80)

from .data import MetaDataLoader
from .config import Config as cfg
from .utils import *


with open(join(dirname(__file__),
               '../data/essentia_feat_columns.pkl'), 'rb') as f:
    ESS_COLS = pkl.load(f)


class BaseAGF:
    """ Artist Group Factor Extractor """
    def __init__(self, metadata_fn, verbose=False):
        """"""
        # load variables
        self.verbose = verbose

        # load metadata
        self.metadata = MetaDataLoader(metadata_fn)

    def process(self):
        """"""
        raise NotImplementedError()

    def _learn_dict(self, fns):
        """"""
        raise NotImplementedError()

    @staticmethod
    def _load_data(fn):
        """"""
        raise NotImplementedError()

    @staticmethod
    def _aggregate_data(batch):
        """"""
        raise NotImplementedError()

    def _learn_factor_model(self, codebook):
        """"""
        # train the top-level factor model with LDA
        lda = LatentDirichletAllocation(cfg.R, n_jobs=cfg.NJOBS)

        # get agf
        z = lda.fit_transform(codebook)

        # register factor model to the instance
        self.fac = lda

        return z


class DictionaryLearningAGF(BaseAGF):
    """"""
    def __init__(self, metadata_fn, batch_size, dict_model,
                 normalize=True, verbose=False):
        """"""
        super().__init__(metadata_fn, verbose=verbose)
        self.batch_size = batch_size
        self.dict_model = dict_model
        self.normalize = normalize

    def process(self, feature_fns):
        """
        Args:
            feature_fns (list of str): list of filenames of feature
        """
        # initiate dictionary model
        self._learn_dict(feature_fns)
        codes = self._build_code(feature_fns)
        z = self._learn_factor_model(codes)
        return z
        
    def _learn_dict(self, fns):
        """"""
        dic = self.dict_model(cfg.K)
        iterator = range(0, len(fns), self.batch_size)
        if self.verbose: iterator = tqdm(iterator, ncols=80)
        for i in iterator:
            batch = []
            for fn in fns[i:i + self.batch_size]:
                batch.append(self._load_data(fn))
            dic.partial_fit(self._aggregate_data(batch))

        # register model
        self.dic = dic

    def _build_code(self, fns):
        """"""
        # binarize the features based on the trained dictionary model
        # cache some useful infos...
        tid2fn = {int(basename(fn).split('.')[0]):fn for fn in fns}
        aid_hash = {
            v:k for k, v in enumerate(
                self.metadata.metadata['artist', 'id'].unique())
        }
        i, j, v = [], [], []  # containors for artist_id, track_id, count
        for artist_id, track_ids in tqdm80(
                self.metadata.artist_audio_map.items()):
            # run over songs from the artist
            data = [
                self._load_data(tid2fn[track_id])
                for track_id in track_ids
                if track_id in tid2fn
            ]
            n_tracks = len(data)
            data = self._aggregate_data(data)
            # check if there's no data
            if data.shape[0] > 0:
                # train the dictionary model
                for k, c in Counter(self.dic.predict(data)).items():
                    i.append(aid_hash[artist_id])
                    j.append(k)

                    if self.normalize:
                        v.append(c / n_tracks)
                    else:
                        v.append(c)

        # build sparse matrix to get the artist BoW
        codes = sp.coo_matrix((v, (i, j)),
                              shape=(len(aid_hash), cfg.K)).tocsr()
        return codes


class MFCCAGF(DictionaryLearningAGF):
    """"""
    def __init__(self, metadata_fn, batch_size=256, normalize=True,
                 verbose=False):
        """"""
        super().__init__(metadata_fn, batch_size, dict_model=MiniBatchKMeans,
                         normalize=normalize, verbose=verbose)

    @staticmethod
    def _load_data(fn):
        """"""
        return np.load(fn)

    @staticmethod
    def _aggregate_data(batch):
        """"""
        return np.concatenate(batch, 0)


class DMFCCAGF(DictionaryLearningAGF):
    """"""
    def __init__(self, metadata_fn, batch_size=256, normalize=True, 
                 verbose=False):
        """"""
        super().__init__(metadata_fn, batch_size, dict_model=MiniBatchKMeans,
                         normalize=normalize, verbose=verbose)

    @staticmethod
    def _load_data(fn):
        """"""
        m = np.load(fn)
        return m[1:] - m[:-1]

    @staticmethod
    def _aggregate_data(batch):
        """"""
        return np.concatenate(batch, 0)


class EssentiaAGF(DictionaryLearningAGF):
    """"""
    def __init__(self, metadata_fn, batch_size=2048, normalize=True,
                 verbose=False):
        """"""
        global ESS_COLS
        super().__init__(metadata_fn, batch_size, dict_model=MiniBatchKMeans,
                         normalize=normalize, verbose=verbose)
        self.cols = ESS_COLS

    @staticmethod
    def _load_data(fn):
        """"""
        with open(fn) as f:
            data = json.load(f)
        return data

    def _aggregate_data(self, batch):
        """"""
        return pd.DataFrame(batch, columns=self.cols)


class SubGenreAGF(BaseAGF):
    """"""
    def __init__(self, metadata_fn, normalize=True, verbose=False):
        """"""
        super().__init__(metadata_fn, verbose=verbose)
        self.normalize = normalize
    
    def process(self):
        """
        Args:
            feature_fns (list of str): list of filenames of feature
        """
        # initiate dictionary model
        codes = self._build_code()
        z = self._learn_factor_model(codes)
        return z

    def _build_code(self):
        """"""
        genres = self.metadata.metadata['track', 'genres'].apply(eval)
        # build the set of genres
        genres2idx = {
            v:k for k, v
            in enumerate(set(chain.from_iterable(genres.values)))
        } 
        tid2genres = genres.to_dict()

        # binarize the features based on the trained dictionary model
        # cache some useful infos...
        aid_hash = {
            v:k for k, v in enumerate(
                self.metadata.metadata['artist', 'id'].unique())
        }
        i, j, v = [], [], []  # containors for artist_id, track_id, count
        for artist_id, track_ids in tqdm80(
                self.metadata.artist_audio_map.items()):

            # run over songs from the artist
            data = list(chain.from_iterable(
                [
                    [genres2idx[g] for g in tid2genres[tid]]
                    for tid in track_ids
                ]
            ))
            n_tracks = len(track_ids)
            # check if there's no data
            if len(data) > 0:
                # train the dictionary model
                for k, c in Counter(data).items():
                    i.append(aid_hash[artist_id])
                    j.append(k)

                    if self.normalize:
                        v.append(c / n_tracks)
                    else:
                        v.append(c)

        # build sparse matrix to get the artist BoW
        codes = sp.coo_matrix((v, (i, j)),
                              shape=(len(aid_hash), cfg.K)).tocsr()
        return codes


