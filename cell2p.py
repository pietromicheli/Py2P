import numpy as np
import random
from scipy.integrate import trapz
from scipy.stats import ttest_ind

from Py2P.utils import *

class Cell2p:

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

        # reference data from rec object
        self.Fraw = rec.Fraw[id]
        self.Fneu = rec.Fraw[id]
        self.spks = rec.spks[id]
        self.params = rec.params

        # refrence usefull sync attibutes
        self.sync = rec.sync
        # self.stims_names = rec.sync.stims_names
        # self.trials_names = rec.sync.trials_names

        # subtract neuropil signal to raw signal
        self.FrawCorr = self.Fraw - self.Fneu * self.params["neuropil_corr"]
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

            self.mean_baseline = np.mean(self.FrawCorr[: self.sync.sync_frames[0]])
            self.dff = (self.FrawCorr - self.mean_baseline) / self.mean_baseline
            self.dff_baseline = self.dff[: self.sync.sync_frames[0]]

        # z-score and filter spikes
        self.zspks = np.where(
            abs(z_norm(self.spks)) < self.params["spks_threshold"], 0, z_norm(self.spks)
        )

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

    def _compute_rmi_(self, a, b, mode="auc"):

        if mode == "auc":

            a_ = trapz(abs(a), dx=1)
            b_ = trapz(abs(b), dx=1)

        elif mode == "peak":

            a_ = np.max(abs(a))
            b_ = np.max(abs(b))

        rmi = (a_ - b_) / (a_ + b_)

        return rmi

    def _compute_snr_imp_(self, a, b):

        """
        Extract noise and signal components from each signal,
        compute SNR and return ration between SNRs
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

            for trial_type in list(self.sync.sync_ds[stim].keys())[
                :-1
            ]:  # last item is stim_window

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

                    # smooth with lp filter
                    # resp_dff = filter(resp_dff, self.params["lowpass_wn"])

                    trials_dff.append(resp_dff)

                    # spiking activity
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

                # convert to array
                trials_dff = np.array(trials_dff)
                trials_spks = np.array(trials_spks)
                trials_zspks = np.array(trials_zspks)

                # calculate QI over df/f traces
                # PENDING: implementation of ttest-based qi
                qi = self._compute_QI_(z_norm(filter(trials_dff, 0.3)))

                if self.params["qi_metrics"]==0:
                
                    if qi > best_qi:
                        best_qi = qi

                else:

                    if qi < best_qi:
                        best_qi = qi

                on = self.params["pre_trial"]
                off = self.params["pre_trial"] + trial_len

                on = self.params["pre_trial"]
                off = self.params["pre_trial"] + trial_len

                analyzed_trials[stim] |= {
                    trial_type: {
                        "trials_dff": trials_dff,
                        "average_dff": np.mean(trials_dff, axis=0),
                        "std_dff": np.std(trials_dff, axis=0),
                        "average_spks": np.mean(trials_spks, axis=0),
                        "std_spks": np.std(trials_spks, axis=0),
                        "average_zspks": np.mean(trials_zspks, axis=0),
                        "std_zspks": np.std(trials_zspks, axis=0),
                        "QI": qi,
                        "window": (on, off),
                    }
                }

        self.analyzed_trials = analyzed_trials
        self.qi = best_qi
        self.is_responsive()

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
                responsive = any(qi >= self.params["qi_threshold"] for qi in qis_stims)

            else:
                responsive = any(qi <= 0.05 for qi in qis_stims)

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

    def calculate_modulation(self, stim, trial_name_1, trial_name_2, mode="rmi"):

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
        """

        if not self.analyzed_trials:

            self.analyze()

        average_resp_1 = self.analyzed_trials[stim][trial_name_1]["average_dff"]
        average_resp_2 = self.analyzed_trials[stim][trial_name_2]["average_dff"]

        if mode == "rmi":

            mod = self._compute_rmi_(average_resp_1, average_resp_2)

        elif mode == "snr":

            mod = self._compute_snr_imp_(average_resp_1, average_resp_2)

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

