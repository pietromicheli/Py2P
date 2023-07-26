from scipy.signal import convolve
from scipy.optimize import curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import numpy as np
from Py2P.core import *

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