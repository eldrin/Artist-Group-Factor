class Config:
    """ General Configuration """
    SR = 22050   # audio samplerate
    WINSZ = 2048 # window size (same for NFFT)
    HOPSZ = 1024 # hop size for STFT anaylsis
    M = 128      # num. of mel bands
    K = 2048     # num. of latent feature K-Means clusters
    R = 40       # num. of latent class for LDA
    D = 25       # num. of coefficients for MFCC
    C = 12       # num. of coefficients for Chroma
    NJOBS = None    # num. of thread for LDA training (-1 means all cores)
