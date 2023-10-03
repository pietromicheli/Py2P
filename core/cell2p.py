import numpy as np
import random
from scipy.integrate import trapz
from scipy.stats import ttest_ind

from Py2P.utils import *

class C2p:

    """
    A Cell object to process, analyze and compute statistics
    on a raw fluoressence trace extracted by suite2p from a single ROI.
    The input Rec2P object will not be copied, just referenced.

    """

    def __init__(self, rec, id: int):

        self.id = id
        self.label = None # usefull for pop analysis
        self.responsive = None
        self.analyzed_trials = None

        self.params = rec.params
        self.sync = rec.sync

        # reference data from rec object
        self.Fraw = rec.Fraw[id]

        if isinstance(rec.Fneu,np.ndarray): 
            self.Fneu = rec.Fneu[id]
        else : self.Fneu = None

        if isinstance(rec.spks,np.ndarray): 
            self.spks = rec.spks[id]
            # z-score and filter spikes
            self.zspks = np.where(
            abs(z_norm(self.spks)) < self.params["spks_threshold"], 0, z_norm(self.spks)
        )
        else : self.spks = None

        # subtract neuropil signal to raw signal if Fneu exists
        if isinstance(self.Fneu,np.ndarray):
            self.FrawCorr = self.Fraw - self.Fneu * self.params["neuropil_corr"]
        else:
            self.FrawCorr = self.Fraw

        # lowpass-filter raw
        self.FrawCorr = filter(self.FrawCorr, self.params["lowpass_wn"])

        # calculate dff on the whole recording
        if self.params["baseline_indices"] != None:

            self.mean_baseline = np.mean(
                self.FrawCorr[
                    self.params["baseline_indices"][0] : self.params["baseline_indices"][1]
                ]
            )

            self.dff = (self.FrawCorr - self.mean_baseline) / self.mean_baseline

            self.dff_baseline = self.dff[
                self.params["baseline_indices"][0] : self.params["baseline_indices"][1]
            ]

        else:
            # just take the mean of the recording
            self.mean_baseline = np.median(self.FrawCorr)
            self.dff = (self.FrawCorr - self.mean_baseline) / self.mean_baseline
            self.dff_baseline = self.dff[: self.sync.sync_frames[0]]

    def _compute_QI_(self, trials: np.ndarray):

        """
        Calculate response quality index as defined by
        Baden et al. over a matrix with shape (reps,time).
        """
        if self.params["qi_metrics"] == 0:
                
            a = np.var(trials.mean(axis=0))
            b = np.mean(trials.var(axis=1))
            return a / b
        
        else:

            n = trials.shape[0]
            mean = np.mean(trials, axis=0)
            pre = mean[:self.params["pre_trial"]]
            post = mean[self.params["pre_trial"]:]
            pvalue = ttest_ind(pre,post)[1]
            # adjust pvalue using Sidak correction
            pvalue_corr = 1-(1-pvalue)**n
            return pvalue_corr

    def _compute_rmi_(self, a, b, mode_rmi="auc"):

        if mode_rmi == "auc":

            a_ = trapz(abs(a), dx=1)
            b_ = trapz(abs(b), dx=1)

        elif mode_rmi == "peak":

            a_ = np.max(a)
            b_ = np.max(b)

        rmi = (a_ - b_) / (a_ + b_)

        return rmi, a_, b_

    def _compute_snr_imp_(self, a, b):

        """
        Extract noise and signal components from each signal,
        compute SNR and return ratio between SNRs
        """

        a_s = filter(a, 0.2, btype="low")
        a_n = filter(a, 0.2, btype="high")
        snr_a = abs(np.mean(a_s) / np.std(a_n))

        b_s = filter(b, 0.2, btype="low")
        b_n = filter(b, 0.2, btype="high")
        snr_b = abs(np.mean(b_s) / np.std(b_n))

        return snr_a / snr_b, snr_a, snr_b

    def analyze(self):

        """
        Compute average responses (df/f and z-scored spkiking activity)
        over trials for each stimulation type defined by the Sync object in rec.
        Return a dictionary with the following structure:
        {
            stim_0:{

                trial_type_0:{

                    "trials_dff",
                    "average_dff",
                    "std_dff",
                    "average_zspks",
                    "std_zspks",
                    "QI",
                    "window",
                }
                ...
            }
            ...
        }
        """

        analyzed_trials = {}

        if self.params["qi_metrics"]==0: best_qi = 0
        else: best_qi = 1

        for stim in self.sync.stims_names:

            analyzed_trials |= {stim: {}}

            if self.params["baseline_extraction"] == 1:

                # extract only one baseline for each stimulus for computing df/f
                mean_baseline = np.mean(
                    self.FrawCorr[
                        self.sync.sync_ds[stim]["stim_window"][0]
                        - self.params["baseline_frames"] : self.sync.sync_ds[stim]["stim_window"][0]
                    ]
                )

            for trial_type in list(self.sync.sync_ds[stim].keys())[:-1]:  # last item is stim_window

                trials_raw = []
                trials_dff = []
                trials_spks = []
                trials_zspks = []

                trial_len = self.sync.sync_ds[stim][trial_type]["trial_len"]
                pause_len = self.sync.sync_ds[stim][trial_type]["pause_len"]

                if pause_len > self.params["max_aftertrial"]:

                    pause_len = self.params["max_aftertrial"]

                for trial in self.sync.sync_ds[stim][trial_type]["trials"]:

                    if self.params["baseline_extraction"] == 0:

                        # extract local baselines for each trial for computing df/f
                        mean_baseline = np.mean(
                            self.FrawCorr[
                                trial[0] - self.params["baseline_frames"] : trial[0]
                            ]
                        )

                    resp = self.FrawCorr[
                        trial[0]
                        - self.params["pre_trial"] : trial[0]
                        + trial_len
                        + pause_len
                    ]

                    resp_dff = (resp - mean_baseline) / mean_baseline

                    trials_raw.append(resp)
                    trials_dff.append(resp_dff)

                    # spiking activity
                    if isinstance(self.spks,np.ndarray):
                        resp_spks = self.spks[
                            trial[0]
                            - self.params["pre_trial"] : trial[0]
                            + trial_len
                            + pause_len
                        ]

                        # z-scored spiking activity
                        resp_zspks = self.zspks[
                            trial[0]
                            - self.params["pre_trial"] : trial[0]
                            + trial_len
                            + pause_len
                        ]

                        trials_spks.append(resp_spks)
                        trials_zspks.append(resp_zspks)

                # statistics on dff amd fraw
                trials_raw = np.array(trials_raw)
                trials_dff = np.array(trials_dff)

                if trials_raw.shape[0] > 1:

                    trials_fraw_avg = np.mean(trials_raw, axis=0)
                    trials_fraw_std = np.std(trials_raw, axis=0)
                    trials_dff_avg = np.mean(trials_dff, axis=0)
                    trials_dff_std = np.std(trials_dff, axis=0)

                else:

                    trials_fraw_avg = trials_raw[0]
                    trials_fraw_std = 0
                    trials_dff_avg = trials_dff[0]
                    trials_dff_std = 0

                # statistics on spks and zspks
                if isinstance(self.spks,np.ndarray) :

                    trials_spks = np.array(trials_spks)
                    trials_zspks = np.array(trials_zspks)

                    if trials_spks.shape[0] > 1:

                        trials_spks_avg = np.mean(trials_spks, axis=0)
                        trials_spks_std = np.std(trials_spks, axis=0)
                        trials_zspks_avg = np.mean(trials_zspks, axis=0)
                        trials_zspks_std = np.std(trials_zspks, axis=0)

                    else:
                        trials_spks_avg = trials_spks[0]
                        trials_spks_std = 0
                        trials_zspks_avg = trials_zspks[0]
                        trials_zspks_std = 0
                    
                else:
                    trials_spks_avg = None
                    trials_spks_std = None
                    trials_zspks_avg = None
                    trials_zspks_std = None

                # calculate QI over df/f traces
                # PENDING: implementation of ttest-based qi
                if trials_dff.shape[0] > 1:
                    qi = self._compute_QI_(z_norm(filter(trials_dff, 0.3)))

                    if self.params["qi_metrics"]==0:
                    
                        if qi > best_qi:
                            best_qi = qi
                    else:

                        if qi < best_qi:
                            best_qi = qi
                else: qi = None

                on = self.params["pre_trial"]
                off = self.params["pre_trial"] + trial_len

                on = self.params["pre_trial"]
                off = self.params["pre_trial"] + trial_len

                analyzed_trials[stim] |= {
                    trial_type: {
                        "average_raw":trials_fraw_avg,
                        "std_raw": trials_fraw_std,
                        "trials_dff": trials_dff,
                        "average_dff": trials_dff_avg,
                        "std_dff": trials_dff_std,
                        "average_spks": trials_spks_avg,
                        "std_spks": trials_spks_std,
                        "average_zspks": trials_zspks_avg,
                        "std_zspks": trials_zspks_std,
                        "QI": qi,
                        "window": (on, off),
                    }
                }

        self.analyzed_trials = analyzed_trials

        if best_qi != 0:
            self.qi = best_qi
            self.is_responsive()
        else:
            # if was impossible to compute QI because the data contain only
            # a single trial for every stimulus, assume is responsive
            self.qi = None
            self.responsive = True 

        return analyzed_trials

    def is_responsive(self):

        """
        Asses responsiveness according to the QI value and the criteria specified in params
        """

        if self.params["resp_criteria"] == 1:

            qis_stims = []

            for stim in self.analyzed_trials:

                qis_trials = []

                for trial_name in self.analyzed_trials[stim]:

                    qis_trials.append(self.analyzed_trials[stim][trial_name]["QI"])

                # for each stimulus, consider only the highest QI calculated
                # over all the different trial types (i.e. Ipsi, Both, Contra)
                qis_stims.append(max(qis_trials))

            if self.params["qi_metrics"]==0:
                responsive = all([qi >= self.params["qi_threshold"] for qi in qis_stims])

            else:
                responsive = all([qi <= 0.05 for qi in qis_stims])

        else:

            if self.params["qi_metrics"]==0:
                responsive = (self.qi >= self.params["qi_threshold"])

            else:
                responsive = (self.qi <= 0.05)


        # # evaluate the responsiveness according to the criteria defined in the config file
        # if self.params["resp_criteria"] == 1:

        #     responsive = all(qi >= qi_threshold for qi in qis_stims)

        # else:

        #     responsive = any(qi >= qi_threshold for qi in qis_stims)

        self.responsive = responsive

        return responsive

    def calculate_modulation(self, stim, trial_name_1, trial_name_2, mode="rmi", slice=(0,-1), **kwargs):

        """
        Calculate Response Modulation Index on averaged responses to
        trial_type_1 vs trial_type_2 during stimulus stim.

        - stim: str
            stimulus name
        - trial_type_1: str
            first trial_type name (i.e "BOTH","CONTRA","IPSI")
        - trial_type_2: str
            second trial_type name (i.e "BOTH","CONTRA","IPSI"),
            it is supposed to be different from trial_type_1.
        - slice: tuple
            slice indexes to use for extracting the portion of the traces that will
            be used for calculating the modulation.
        """

        if not self.analyzed_trials:

            self.analyze()

        average_resp_1 = self.analyzed_trials[stim][trial_name_1]["average_dff"][slice[0]:slice[1]]
        average_resp_2 = self.analyzed_trials[stim][trial_name_2]["average_dff"][slice[0]:slice[1]]


        if mode == "rmi":

            mod = self._compute_rmi_(average_resp_1, average_resp_2, **kwargs)

        elif mode == "snr":

            mod = self._compute_snr_imp_(average_resp_1, average_resp_2, **kwargs)

        return mod

    def calculate_random_modulation(self, stim, trial_name, n_shuff=100, mode="rmi"):

        """
        Quintify the intrinsic variability in the responses generated by
        the same trial type, i.e. only due to stochastic effects, using RMI.

        - stim: str
            name of the stimulus type
        - trial_name: str
            the trial type for which the intrinsic variability is quantified.
        - n_shuff: int
            number of shuffling
        """

        shuff_mods = []

        for i in range(n_shuff):

            n_trials = len(self.sync.sync_ds[stim][trial_name]["trials"])

            # generate two random groups of responses generate by trial_name
            shuff_trials_idx = np.arange(n_trials)
            random.shuffle(shuff_trials_idx)

            shuff_trials_idx_a = shuff_trials_idx[: int(n_trials / 2)]
            shuff_trials_idx_b = shuff_trials_idx[int(n_trials / 2) :]

            trials_a = []

            for trial in shuff_trials_idx_a:

                trials_a.append(
                    self.analyzed_trials[stim][trial_name]["trials_dff"][trial]
                )

            trials_b = []

            for trial in shuff_trials_idx_b:

                trials_b.append(
                    self.analyzed_trials[stim][trial_name]["trials_dff"][trial]
                )

            shuff_mods.append(
                self._compute_rmi_(np.mean(trials_a, axis=0), np.mean(trials_b, axis=0))
            )

            # for trial_a, trial_b in zip(shuff_trials_idx_a,shuff_trials_idx_b):

            #     if mode=='rmi':

            #         shuff_mods.append(
            #             self._compute_rmi_(self.analyzed_trials[stim][trial_name]["trials_dff"][trial_a],
            #                                 self.analyzed_trials[stim][trial_name]["trials_dff"][trial_b])
            #         )

            #     elif mode=='snr':

            #         shuff_mods.append(
            #             self._compute_snr_imp_(self.analyzed_trials[stim][trial_name]["trials_dff"][trial_a],
            #                                 self.analyzed_trials[stim][trial_name]["trials_dff"][trial_b])
            #         )

        shuff_controls = np.mean(shuff_mods)

        return shuff_controls

