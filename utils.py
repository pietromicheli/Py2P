import os
import shutil
from scipy.signal import butter, filtfilt, sosfiltfilt, lfilter
from scipy.optimize import curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from Py2P.core import CONFIG_FILE_TEMPLATE

#############################
###---UTILITY FUNCTIONS---###
#############################


def generate_params_file():

    """
    Generate a parameters file in the current working dir.
    This file contains a list of all the parameters that will used for the downsteream analysis,
    set to a default value.
    """

    files = os.listdir(os.getcwd())

    if "params.yaml" not in files:

        print("> Config file generated. All parameters set to default.")

        return shutil.copy(CONFIG_FILE_TEMPLATE, "params.yaml")

    else:

        print("> Using the parameters file found in data_path.")

        return "params.yaml"


def filter(s, wn, ord=6, btype="low", analog=False, fs=None, mode="filtfilt"):

    """
    Apply scipy's filtfilt to signal s.
    """

    b, a = butter(ord, wn, btype, analog, fs=fs)

    if mode == "filtfilt":
        s_filtered = filtfilt(b, a, s)
    elif mode == "sosfiltfilt":
        s_filtered = sosfiltfilt(b, a, s)
    elif mode == "lfilter":
        s_filtered = lfilter(b, a, s)

    return s_filtered


def z_norm(s, include_zeros=False):

    """
    Compute z-score normalization on signal s
    """

    if not isinstance(s, np.ndarray):

        s = np.array(s)

    if include_zeros:

        if len(s.shape)==1:

            s_mean = np.mean(s)
            s_std = np.std(s)
            return (s - s_mean) / s_std
        
        else:

            s_mean = np.mean(s,axis=1)
            s_std = np.std(s,axis=1)
            return ((s.T - s_mean) / s_std).T

    elif s[s != 0].shape[0] > 1: ### Needs to be fixed!

        if len(s.shape)==1:

            s_mean = np.mean(s[s != 0])
            s_std = np.std(s[s != 0])
            return (s - s_mean) / s_std
        
        else:
            
            s_mean = np.nanmean(np.where(s!=0,s,np.nan),1)
            s_std = np.nanstd(np.where(s!=0,s,np.nan),1)
            return ((s.T - s_mean) / s_std).T

    return np.zeros(s.shape)


def lin_norm(s, lb=0, ub=1):

    """
    Compute linear normalization between lb and ub on input signal s
    """
    return (ub - lb) * ((s - s.min()) / (s.max() - s.min())) + lb


def TSNE_embedding(data=None, **kwargs):

    if len(data) < 50:
        n_comp = len(data)
    else:
        n_comp = 50

    if kwargs:
        tsne_params = kwargs
    else:
        tsne_params = {
            "n_components": 2,
            "verbose": 1,
            "metric": "cosine",
            "early_exaggeration": 4,
            "perplexity": 15,
            "n_iter": 2000,
            "init": "pca",
            "angle": 0.1,
        }
    # run PCA
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(data)
    # run t-SNE
    tsne = TSNE(**tsne_params)

    transformed = tsne.fit_transform(transformed)

    return transformed


def k_means(data, n_clusters):

    # run Kmeans
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", algorithm="auto").fit(data)

    return kmeans.labels_


def GMM(data, **kwargs):

    # run Gaussian Mixture Model
    gmm = BayesianGaussianMixture(**kwargs)
    gm_labels = gmm.fit_predict(data)

    print(gmm.converged_)
    return gm_labels


def find_optimal_kmeans_k(x):

    """
    Find the optimal number of clusters to use for k-means clustering on x.
    """

    # find optimal number of cluster for Kmeans
    Sum_of_squared_distances = []

    K = range(1, 8)

    for k in K:

        kmeans = KMeans(n_clusters=k, init="random").fit(x)

        Sum_of_squared_distances.append(kmeans.inertia_)

        labels = kmeans.labels_

    def _monoExp_(x, m, t, b):
        return m * np.exp(-t * x) + b

    x = np.arange(1, 8)

    p0 = (200, 0.1, 50)  # start with values near those we expect

    p, cv = curve_fit(_monoExp_, x, Sum_of_squared_distances, p0)

    m, t, b = p

    x_plot = np.arange(1, 8, 0.01)

    fitted_curve = _monoExp_(x_plot, m, t, b)

    # find the elbow point
    xx_t = np.gradient(x_plot)

    yy_t = np.gradient(fitted_curve)

    curvature_val = (
        np.abs(xx_t * fitted_curve - x_plot * yy_t)
        / (x_plot * x_plot + fitted_curve * fitted_curve) ** 1.5
    )

    dcurv = np.gradient(curvature_val)

    elbow = np.argmax(dcurv)

    return round(x_plot[elbow])


def check_len_consistency(sequences, mode='trim', mean_len=10):

    """
    Utility function for correcting for length inconsistency in a list of 1-D iterables.

    - sequence (list of iterables):
        list of array-like elements that will be trimmed to the same (minimal) length.
    - mode (str):
        'trim': finds the size of the shortes iterable and trim the other iterables accordingly.
        'pad': finds the size of the longest iterable and mean-pad the other iterables accordingly.
    - mean_len (int):
        number of values used to calculate the mean used for padding if mode='pad'

    """

    # find minimal length

    lengths = [len(sequence) for sequence in sequences]

    if mode=='trim':
        
        min_len = np.min(lengths)

        # trim to minimal length

        sequences_new = []

        for seq in sequences:

            sequences_new.append(seq[:min_len])

    elif mode=='pad':

        max_len = np.max(lengths)

        # trim to minimal length

        sequences_new = []

        for seq in sequences:

            sequences_new.append(np.pad(seq,(len(seq)-max_len),mode='mean',stat_length=mean_len))

    return sequences_new


def fit_oscillator(s,fs,plot_ps=False,freq_int=[0.2,0.2]):

    def optSinWrap(freq_osc=None):
        def optSin(x,freq_osc,phi):

            return np.sin(x*2*np.pi*freq_osc + phi)
        return optSin

    dt = 1 / fs
    #normalize signal
    sNorm = lin_norm(s,-1,1)
    # guess foundamental frequency of oscillation

    # OLD #
    # s_ps = np.abs(np.fft.fft(s))**2
    # freqs = np.fft.fftfreq(s.size,dt)
    # freq_osc = abs(freqs[np.argmax(s_ps[1:])+1])

    s_ps = np.abs(rfft(s))**2
    freqs = rfftfreq(s.size,dt)
    freq_osc = abs(freqs[np.argmax(s_ps[1:])+1])

    # le and ue are lower and upper edges of the frequencies spike which peak at freq_osc
    s_ps_diff = np.diff(s_ps[1:])
    le = freq_osc-freq_int[0] # freqs[np.argmax(s_ps_diff)+1]
    ue = freq_osc+freq_int[1] # freqs[np.argmin(s_ps_diff)+1]
    freq_range = np.array([le,ue])

    # idx = np.argsort(freqs) #useless

    if plot_ps:
        plt.figure()
        plt.title('{}Hz'.format(round(freq_osc,2)))
        plt.plot(freqs[:-1], np.diff(s_ps))
        # plt.plot(np.flip(freqs[:-1]), np.diff(np.flip(s_ps)))
        plt.plot(freqs, s_ps)
        plt.scatter(freq_osc,s_ps[1:].max(),c='r')
        plt.scatter(le,s_ps[np.argmax(s_ps_diff)+1],c='g')
        plt.scatter(ue,s_ps[np.argmin(s_ps_diff)+3],c='g')
        plt.show()

    # filter and normalize
    sfilt = filter(sNorm,freq_osc,btype='lp',fs=fs)
    sfiltNorm = lin_norm(sfilt,-1,1)

    # define time
    t = np.linspace(0,len(s)/fs,len(s))

    # optimize
    optSin=optSinWrap()
    popt = curve_fit(optSin,t,sfiltNorm,p0=[freq_osc,0],
                    bounds=([freq_osc-0.2,-np.pi],[freq_osc+0.2,np.pi]))[0]
    
    print('Freq peak:{}, Freq_fit:{}, Phase_opt:{}'.format(freq_osc,popt[0],popt[1]))

    return {'f':freq_osc,'fopt':popt[0],'phi':popt[1],'freq_range':freq_range},optSin(t,popt[0],popt[1])


def kill_freq(s,fs,f,freq_range,type='bs',lpcut=1.2):

    dt = 1/fs

    yf = rfft(s)
    W = rfftfreq(s.size, d=dt)
    cut_f = yf.copy()
    # kill freqs
    if type=='bs':
        cut_f[np.where((W>=freq_range[0])&(W<=freq_range[1]))] = 0 
        
    elif type=='bp':
        cut_f[np.where((W<=freq_range[0])|(W>=freq_range[1]))] = 0  

    cut_s = irfft(cut_f)
    # lowpass
    cut_s = filter(cut_s,lpcut,fs=fs)

    return cut_s


def deoscillate(x,x_train,fs,lpcut=1.2,norm=True,plot=False,freq_int=[0.2,0.2]):

    # estimate oscillation frequency from x_train
    # tosc = np.linspace(0,x_train.size/fs,x_train.size)
    if norm:
        x_train = lin_norm(x_train,-1,1)

    osc_params,osc_fit = fit_oscillator(x_train,fs,plot_ps=plot,freq_int=freq_int)

    x_filt = kill_freq(x,fs,osc_params['f'],osc_params['freq_range'])
    
    return x_filt