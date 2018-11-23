import pandas as pd


class MetaDataLoader:
    """"""
    def __init__(self, metadata_fn):
        """"""
        self.metadata_fn = metadata_fn
        self.metadata = pd.read_csv(self.metadata_fn,
                                    header=[0, 1], index_col=0)
        # retrieve medium subset
        self.metadata = self.metadata[
            (self.metadata['set']['subset'] == 'medium') |
            (self.metadata['set']['subset'] == 'small')
        ]

    @property
    def track_artist_map(self):
        """"""
        return dict(self.metadata['artist', 'id'].items())

    @property
    def track_genre_map(self):
        """"""
        return dict(self.metadata['track', 'genre_top'].items())

    @property
    def track_subgenre_map(self):
        """"""
        return dict(map(lambda kv: (kv[0], eval(kv[1])),
                        self.metadata['track', 'genres_all'].items()))

    @property
    def artist_audio_map(self):
        """"""
        return dict(
            self.metadata.reset_index()
                .groupby(('artist', 'id'))
                .track_id.apply(list).items()
        )
