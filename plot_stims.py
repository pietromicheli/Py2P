import numpy as np
from scipy.special import sinc
from scipy.signal import chirp

def chirp_stim(
        ax=None,
        pad_l = 40,
        stim_len = None,
        freq_cr = 2,#0.55,
        fr = 15.5):

    pad_left = np.ones(pad_l)*0.5
    chunk_OFF1 = np.zeros(int(2*fr))
    chunk_ON = np.ones(int(3*fr))
    chunk_OFF2 = np.zeros(int(3*fr))
    BG = np.ones(int(2*fr))*0.5
    t_fm = np.linspace(0,20,int(8*fr))
    FM = (chirp(t_fm,0.2,40,1)+1)/2
    ttot = 12/freq_cr
    t_am = np.linspace(0,ttot,int(ttot*fr))
    # a_am = np.linspace(0.01,0.5,int(21*fr))
    scale = t_am/ttot
    AM = (np.sin(t_am*2*np.pi*freq_cr))/2*scale+0.5
    
    stim_conc = np.concatenate([pad_left,chunk_OFF1,chunk_ON,chunk_OFF2,chunk_ON*0.5,
                                FM,BG,AM,BG,chunk_OFF1,chunk_ON,chunk_OFF1,chunk_ON,chunk_OFF1])
    
    pad_right = np.ones((stim_len-len(stim_conc)))*0.5
    stim_conc = np.concatenate([stim_conc, pad_right])

    if ax != None:
        ax.plot(stim_conc,c='k',linewidth=2)
        green_x = len(stim_conc)-len(pad_right)-int(8.5*fr)
        blue_x = len(stim_conc)-len(pad_right)-int(3.5*fr)
        ax.axvline(green_x, 0, 1, linewidth=20, color='g',alpha=0.3)
        ax.axvline(blue_x, 0, 1, linewidth=20, color='b',alpha=0.3)
        
        ax.set_xticks(ax.get_xticks()[1:-1], (ax.get_xticks()[1:-1]/fr).astype(int))
        ax.set_ylabel("brightness", fontsize=18)

    return ax,stim_conc

def full_field_stim(
        ax=None,
        pad_l = 40,
        stim_len = None,
        fr = 15.5):


    pad_left = np.zeros(pad_l)+0.25 # 50% contrast
    chunk_1 = np.ones(int(5*fr))*0.55

    stim_conc = np.concatenate([pad_left,chunk_1])

    pad_right = np.zeros((stim_len-len(stim_conc)))+0.25

    stim_conc = np.concatenate([stim_conc, pad_right])  

    stim_start = len(stim_conc)-len(pad_right)-int(5*fr)
    stim_end = len(stim_conc)-len(pad_right)

    if ax != None:
        ax.plot(stim_conc,c='k',linewidth=2)
        ax.set_ylim(0,1)
        ax.axvspan(stim_start, stim_end, color='y', alpha=0.1)
        ax.set_ylabel("brightness", fontsize=18)

    return ax,stim_conc

def contrast_ramp_stim(
    ax=None,
    pad_l = 40,
    stim_len = None,
    fr = 15.5,
    freq = 0.55 #0.57
    ):

   ncycles = 12
   ttot = int(ncycles/freq)

   pad_left = np.ones(pad_l)*0.5
   t_am = np.linspace(0,ttot,int(ttot*fr)) 
   scale = t_am/21
   AM = (np.sin(t_am*2*np.pi*freq))/2*scale+0.5
   
   stim_conc = np.concatenate([pad_left,AM])
   
   pad_right = np.ones((stim_len-len(stim_conc)))*0.5
   stim_conc = np.concatenate([stim_conc, pad_right])

   if ax!=None:
    ax.plot(stim_conc,c='k',linewidth=2)
    
    ax.set_xticks(ax.get_xticks()[1:-1], (ax.get_xticks()[1:-1]/fr).astype(int))
    ax.set_ylabel("brightness", fontsize=18)

   return ax,stim_conc
