import os
import shutil
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.integrate import trapz
import yaml
import warnings
from tqdm import tqdm
import pathlib
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

from Py2P.sync import Sync

pathlib.Path(__file__).parent.resolve()

CONFIG_FILE_TEMPLATE = "%s\\params.yaml" % pathlib.Path(__file__).parent.resolve()

DEFAULT_PARAMS = {}

########################
###---MAIN CLASSES---###
########################

class Rec2P:

    """
    A recording master object
    """

    def __init__(self, data_path: str, sync: Sync):

        """
        Create a Rec2P object from .npy files generated by Suite2P.

        - data_path:
            absolute path to suite2P outoput directory
        - sync:
            S object to use for aligning the stimuli to the recording
        """

        expected_files = [
            "F.npy",
            "Fneu.npy",
            "iscell.npy",
            "spks.npy",
            "stat.npy",
            "ops.npy",
        ]

        files = os.listdir(data_path)

        # check all the files are there
        for f in expected_files:

            if f not in files:

                raise Exception("%s file not found in %s" % (f, data_path))

        # load data
        print("\n> loading data from %s ..." % data_path, end=" ")

        self.data_path = data_path
        self.Fraw = np.load(data_path + "\\F.npy")
        self.Fneu = np.load(data_path + "\\Fneu.npy")
        self.iscell = np.load(data_path + "\\iscell.npy")
        self.spks = np.load(data_path + "\\spks.npy")
        self.stat = np.load(data_path + "\\stat.npy", allow_pickle=True)
        self.ops = np.load(data_path + "\\ops.npy", allow_pickle=True)

        print("OK")

        self.data_path = data_path
        self.params_file = generate_params_file()

        self.sync = sync
        self.cells = None
        self.params = None

        # read parameters from .yaml params file
        self.load_params()

    def get_nframes(self):
        return self.Fraw.shape[1]

    def get_ncells(self):
        return self.Fraw.shape[0]   

    def load_params(self):

        """
        Read parameters from .yaml params file
        """

        with open(self.params_file, "r") as f:

            self.params = yaml.load(f, Loader=yaml.Loader)

            # update also for all the cells
            if self.cells != None:

                for cell in self.cells:

                    self.cells[cell].params = self.params

            print("> parameters loaded.")

        return self.params

    def get_responsive(self):

        """
        Get a list containing the ids of all the responsive cells
        """

        ids = []

        for cell in self.cells:

            if self.cells[cell].responsive:

                ids.append(cell)

        return ids

    def get_cells(
        self, 
        keep_unresponsive: bool = False, 
        n: int = None,
        clean_memory = True
        ):

        """
        Retrive the cells from the recording files.

        - keep_unresponsive: bool
            decide wether to keep also the cells classified as unresponsive using
            the criteria and the threshold specified in thge config file.
        - n: int
            maximum number of cell to extract. If None, all the cells will be extracted/
        - clean_memory: Bool
            if True, delete original close original .npy files
        """

        print("\n> Extracting cells ...")
        cells = {}
        responsive = []

        if n == None:

            n_cells = self.get_ncells()

        else:

            n_cells = n

        for id in tqdm(range(n_cells)):

            if self.params["use_iscell"]:

                # check the ROIS has been classified as cell by suite2p
                if not self.iscell[id][0]:
                    continue

            cell = Cell2P(self, id)

            cell.analyze_trials()

            if not keep_unresponsive and not cell.responsive:

                # discard cell and free memory
                del cell

            else:

                cells |= {id: cell}

                if cell.responsive:

                    responsive.append(id)

        self.cells = cells

        if not cells:

            warnings.warn("No cells found!", RuntimeWarning)

        else:

            print(
                "> %d responsive cells found (tot: %d, keep_unresponsive: %r)"
                % (len(responsive), n_cells, keep_unresponsive)
            )

        if clean_memory:

            del self.Fraw
            del self.Fneu
            del self.spks
            del self.iscell

        return self.cells

    def get_populations(
        self,
        stims_names=None,
        trials_names=None,
        n_clusters=None,
        use_tsne=False,
        type="dff",
        normalize="norm",
        plot=True,
        ):

        """
        Clusterize the activity traces of all the cells into population using PCA and K-means.

        - stims_names: list
            The stimulation conditions to use for extracting the response feature of each cell.
            By default the respopnses to all the stimuli will be used.
        - trials_names: list
            The responses to the specified trials will be concatenated and used as features for
            th PCA, for all the stimuli specified by stims_names.
            By default the respopnses to all the trial types will be used.
        - n_clusters: int
            Number of cluster to use for k means clustering
        - use_tsne: bool
            wether to compute tsne embedding after PCA decomposition
        - type: str
            can be either "dff" or "zspks"
        - normalize: str
            'norm': signals will be normalized between 0 and 1 before running PCA
            'z': signals will be normalized using z score normalization before running PCA
            otherwise, no normalization will be applied

        """

        # check if the cells have already been retrived

        if self.cells == None:

            self.get_cells()

        all_mean_resp = []

        if stims_names == None:

            stims_names = self.sync.stims_names

        responsive = self.get_responsive()

        for cell in responsive:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for stim in stims_names:

                if trials_names == None:

                    trials_names = list(self.sync.sync_ds[stim].keys())[:-1]

                for trial_name in trials_names:

                    r = average_resp[stim][trial_name]["average_" + type]

                    # cut the responses
                    start = average_resp[stim][trial_name]['window'][0]
                    stop = average_resp[stim][trial_name]['window'][1]

                    r = r[start:int(stop+start/2)]
                    # low-pass filter 
                    r = filter(r,0.3)

                    concat_stims = np.concatenate((concat_stims, r))

            if normalize == "norm":

                concat_stims = (concat_stims - concat_stims.min()) / (concat_stims.max() - concat_stims.min())

            elif normalize == "z":

                concat_stims = z_norm(r, True)

            all_mean_resp.append(concat_stims)

        # convert to array
        all_mean_resp = np.array(all_mean_resp)
        x = np.array(all_mean_resp)

        # run PCA and tSNE if desired

        if use_tsne:

            if len(x)<50:
                n_comp = len(x)
            else:
                n_comp = 50
                
            # run PCA
            pca = PCA(n_components=n_comp)
            transformed = pca.fit_transform(x)
            # run t-SNE
            tsne = TSNE(n_components=2, 
                        verbose=1, 
                        metric='cosine', 
                        early_exaggeration=2, 
                        perplexity=15, 
                        n_iter=2000, 
                        init='pca', 
                        angle=1)
            
            transformed = tsne.fit_transform(transformed)
        
        else:

            # run PCA
            pca = PCA(n_components=3)
            transformed = pca.fit_transform(x)

        # if the nuber of cluster is not specified, find optimal n
        if n_clusters == None:

            n_clusters = find_optimal_kmeans_k(transformed)

        # run Kmeans
        kmeans = KMeans(n_clusters=n_clusters, 
                        init="k-means++",
                        algorithm="auto").fit(transformed)
        
        labels = kmeans.labels_

        # retrive clusters
        clusters = []

        for n in np.unique(labels):

            indices = np.squeeze(np.argwhere(labels == n))
            c = []

            for i in indices:

                c.append(responsive[i])

            clusters.append(c)

        if plot:

            clist = list(colors.TABLEAU_COLORS.keys())

            if use_tsne:

                algo = "t-SNE"

                Xax = transformed[:, 0]
                Yax = transformed[:, 1]

                fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111)

                fig.patch.set_facecolor("white")

                for l in np.unique(labels):

                    ix = np.where(labels == l)
                    ax.scatter(
                        Xax[ix], Yax[ix], 
                        c=clist[l],
                        s=50, 
                        marker='o',
                        alpha=0.5
                    )

                ax.set_xlabel("%s 1"%algo, fontsize=9)
                ax.set_ylabel("%s 2"%algo, fontsize=9)
                ax.set_title("%d ROIs"%(len(Xax)))


            else:

                algo = "PCA"

                Xax = transformed[:, 0]
                Yax = transformed[:, 1]
                Zax = transformed[:, 2]

                cdict = {0: "c", 1: "g", 2: "r", 3: "y", 4:"m"}

                fig = plt.figure(figsize=(7, 5))
                # ax = fig.add_subplot(111, projection="3d")
                ax = fig.add_subplot(111)

                fig.patch.set_facecolor("white")

                for l in np.unique(labels):

                    ix = np.where(labels == l)
                    # ax.scatter(
                    #     Xax[ix], Yax[ix], Zax[ix], c=clist[l], s=40)
                    ax.scatter(
                          Xax[ix], Yax[ix], c=clist[l], s=40)

                ax.set_xlabel("%s 1"%algo, fontsize=9)
                ax.set_ylabel("%s 2"%algo, fontsize=9)
                # ax.set_zlabel("%s 3"%algo, fontsize=9)

                ax.set_title("%d ROIs"%(len(Xax)))

                ax.view_init(30, 60)


        return clusters


class Cell2P:

    """
    A Cell object to process, analyze and compute statistics
    on a raw fluoressence trace extracted by suite2p from a single ROI.
    The input Rec2P object will not be copied, just referenced.
    
    """

    def __init__(self, rec: Rec2P, idx: int):

        self.idx = idx
        self.responsive = None
        self.analyzed_trials = None

        # reference data from rec object
        self.Fraw = rec.Fraw[idx]
        self.Fneu = rec.Fraw[idx]
        self.spks = rec.spks[idx]
        self.params = rec.params

        # refrence usefull sync attibutes
        self.sync = rec.sync.sync_ds
        self.stims_names = rec.sync.stims_names
        self.trials_names = rec.sync.trials_names

        # subtract neuropil signal to raw signal
        self.FrawCorr = self.Fraw - self.Fneu * self.params["neuropil_corr"]
        # lowpass-filter raw
        self.FrawCorr = filter(self.FrawCorr, self.params["lowpass_wn"])

        # calculate dff on the whole recording
        if self.params["baseline_indices"] != None:

            self.mean_baseline = np.mean(
                self.FrawCorr[
                    self.params["baseline_indices"][0] : self.params[
                        "baseline_indices"
                    ][1]
                ]
            )
            dff = (self.FrawCorr - self.mean_baseline) / self.mean_baseline

            dff_baseline = dff[
                self.params["baseline_indices"][0] : self.params["baseline_indices"][1]
            ]

        else:

            self.mean_baseline = np.mean(self.FrawCorr[: rec.sync.sync_frames[0]])

            self.dff = (self.FrawCorr - self.mean_baseline) / self.mean_baseline

            self.dff_baseline = self.dff[: rec.sync.sync_frames[0]]

        # low-pass filter dff
        # self.dff = filter(self.dff, self.params["lowpass_wn"])

        # self.dff_baseline = filter(self.dff_baseline, self.params["lowpass_wn"])

    def _compute_QI_(self, trials: np.ndarray):

        """
        Calculate response quality index as defined by Baden et al.
        over a matrix with shape (reps,time).
        """

        a = np.var(trials.mean(axis=0))
        b = np.mean(trials.var(axis=1))

        return a / b

    def _compute_rmi_(self,a, b, mode='auc'):

        if mode == 'auc':

            a_ = trapz(abs(a), dx=1)
            b_ = trapz(abs(b), dx=1)

        elif mode == 'peak':

            a_ = np.max(abs(a))
            b_ = np.max(abs(b))
        

        rmi = (a_ - b_) / (a_ + b_)

        return rmi
    
    def _compute_snr_imp_(self,a, b):

        '''
        Extract noise and signal components from each signal,
        compute SNR and return ration between SNRs
        '''
        
        a_s = filter(a,0.2,btype="low")
        a_n = filter(a,0.2,btype="high")
        snr_a = abs(np.mean(a_s)/np.std(a_n))

        b_s = filter(b,0.2,btype="low")
        b_n = filter(b,0.2,btype="high")
        snr_b = abs(np.mean(b_s)/np.std(b_n))

        return snr_a/snr_b,snr_a,snr_b

    def analyze_trials(self):

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

        for stim in self.stims_names:

            analyzed_trials |= {stim: {}}

            if self.params["baseline_extraction"] == 1:

                # extract only one baseline for each stimulus for computing df/f
                mean_baseline = np.mean(
                    self.FrawCorr[
                        self.sync[stim]["stim_window"][0]
                        - self.params["baseline_frames"] : self.sync[stim][
                            "stim_window"
                        ][0]
                    ]
                )

            for trial_type in list(self.sync[stim].keys())[:-1]:  # last item is stim_window

                trials_dff = []
                trials_spks = []
                trials_zspks = []

                trial_len = self.sync[stim][trial_type]["trial_len"]
                pause_len = self.sync[stim][trial_type]["pause_len"]

                if pause_len > self.params["max_aftertrial"]:

                    pause_len = self.params["max_aftertrial"]

                for trial in self.sync[stim][trial_type]["trials"]:

                    if self.params["baseline_extraction"] == 0:

                        # extract local baselines for each trial for computing df/f
                        mean_baseline = np.mean(
                            self.FrawCorr[
                                trial[0] - self.params["baseline_frames"] : trial[0]
                            ]
                        )

                    resp = self.FrawCorr[
                        trial[0]
                        - self.params["baseline_frames"] : trial[0]
                        + trial_len
                        + pause_len
                    ]

                    resp_dff = (resp - mean_baseline) / mean_baseline

                    # smooth with lp filter
                    # resp_dff = filter(resp_dff, self.params["lowpass_wn"])

                    trials_dff.append(resp_dff)

                    # calculate z-scored spiking activity
                    resp_spks = self.spks[
                        trial[0]
                        - self.params["baseline_frames"] : trial[0]
                        + trial_len
                        + pause_len
                    ]

                    resp_zspks = z_norm(resp_spks)

                    # threshold z-scored spiking activity
                    resp_zspks = np.where(
                        abs(resp_zspks) < self.params["spks_threshold"], 0, resp_zspks
                    )

                    trials_spks.append(resp_spks)
                    trials_zspks.append(resp_zspks)

                # convert to array
                trials_dff = np.array(trials_dff)
                trials_spks = np.array(trials_spks)
                trials_zspks = np.array(trials_zspks)

                # calculate QI over df/f traces
                qi = self._compute_QI_(filter(trials_dff, 0.3))

                on = self.params["baseline_frames"]
                off = self.params["baseline_frames"] + trial_len

                analyzed_trials[stim] |= {
                    trial_type: {
                        "trials_dff": trials_dff,
                        "average_dff": np.mean(trials_dff, axis=0),
                        "std_dff": np.std(trials_dff, axis=0),
                        "average_spks": np.mean(trials_spks, axis=0),
                        "average_zspks": np.mean(trials_zspks, axis=0),
                        "std_zspks": np.std(trials_spks, axis=0),
                        "QI": qi,
                        "window": (on, off),
                    }
                }

        self.analyzed_trials = analyzed_trials
        self.is_responsive()

        return analyzed_trials

    def is_responsive(self, qi_threshold=None):

        """
        Asses responsiveness according to the QI value and the criteria specified in params
        """

        if qi_threshold == None:

            qi_threshold = self.params["qi_threshold"]

        qis_stims = []

        for stim in self.analyzed_trials:

            qis_trials = []

            for trial_name in self.analyzed_trials[stim]:

                qis_trials.append(self.analyzed_trials[stim][trial_name]["QI"])

            # for each stimulus, consider only the highest QI calculated
            # over all the different trial types (i.e. Ipsi, Both, Contra)
            qis_stims.append(max(qis_trials))

        # evaluate the responsiveness according to the criteria defined in the config file
        if self.params["resp_criteria"] == 1:

            responsive = all(qi >= qi_threshold for qi in qis_stims)

        else:

            responsive = any(qi >= qi_threshold for qi in qis_stims)

        self.responsive = responsive

        return responsive

    def calculate_modulation(self, stim, trial_name_1, trial_name_2, mode='rmi'):

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

            self.analyze_trials()

        average_resp_1 = self.analyzed_trials[stim][trial_name_1]["average_dff"]
        average_resp_2 = self.analyzed_trials[stim][trial_name_2]["average_dff"]

        if mode=='rmi':

            mod = self._compute_rmi_(average_resp_1, average_resp_2)

        elif mode=='snr':

            mod = self._compute_snr_imp_(average_resp_1, average_resp_2)

        return mod

    def calculate_random_modulation(self, stim, trial_name, n_shuff=100, mode='rmi'):

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

            n_trials = len(self.sync[stim][trial_name]["trials"])

            # generate two random groups of responses generate by trial_name
            shuff_trials_idx = np.arange(n_trials)
            random.shuffle(shuff_trials_idx)

            shuff_trials_idx_a = shuff_trials_idx[: int(n_trials/2)]
            shuff_trials_idx_b = shuff_trials_idx[int(n_trials/2) :]

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

            shuff_mods.append(self._compute_rmi_(np.mean(trials_a,axis=0),np.mean(trials_b,axis=0)))

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


class Batch2P:

    """
    A class for performing bartch analysis of multiple recordings.
    All the recordings contained in a Batch2P object must share at 
    least one stimulation condidition with at least one common trial 
    type (e.g. CONTRA, BOTH or IPSI)

    """

    def __init__(self, data_dict: dict, groups={}):

        """
        Create a Batch2P object from a data dictionary.

        - data_dict: dict
            A dictionary where each key is an absolute path to the .npy files
            of a recording, and its vlue is a Sync object generated for that recording.
            Something like {data_path_rec0:Sync_rec0,...,data_path_recN:Sync_recN}
            for a Batch2P object containing N independent recordings.
        - groups: dict
            A dictionary for assigning each recording to a group. This is useful for keeping
            joint analysis of recordings performed in different conditions (e.g. control and treated).
            The keys of the dictionary must be the same as data_dict (datapaths), and the values int numbers.
            By thefault, all recordings loaded are assigned to group 0.
        """

        # be sure that the sync object of all the recordings share at least 1 stimulus.
        # only shared stimuli will be used.

        self.stims_trials_intersection = {}

        stims_allrec = [sync.stims_names for sync in data_dict.values()]

        stims_intersection = set(stims_allrec[0])

        for stims in stims_allrec[1:]:

            stims_intersection.intersection(set(stims))

        # also, for all the shared stimuli,
        # select only trials type are shared for that specific stimulus by all recs.

        for stim in stims_intersection:

            # start with trials for stimulus "stim" in first sync object
            all_trials = list(list(data_dict.values())[0].sync_ds[stim].keys())[:-1]

            trials_intersection = set(all_trials)

            for sync in list(data_dict.values())[1:]:

                # last item is "window_len"
                trials_intersection.intersection(set(list(sync.sync_ds[stim].keys())[:-1]))

            self.stims_trials_intersection |= {stim:list(trials_intersection)}

        
        # generate params.yaml
        generate_params_file()

        # instantiate the Rec2P objects
        self.recs = {}

        for rec_id, (data_path, sync) in enumerate(data_dict.items()):

            if data_path not in groups:
                group_id = 0

            else:
                group_id = groups[data_path]

            rec = Rec2P(data_path, sync)
            self.recs |= {rec_id:(rec,group_id)}

        self.cells = None

    def load_params(self):

        """
        Read parameters from .yaml params file
        
        """

        for rec in self.recs:

            rec.load_params()
        
    def get_cells(self):

        """
        Extract all the cells from the individual recordings and assign new ids.
        Id is in the form G_R_C, where G,R and C are int which specify the group,
        the recording and the cell ids.
        
        """

        self.cells = {}

        for (rec_id,value) in self.recs.items():

            rec = value[0]
            group_id = value[1]

            # retrive cells for each recording
            rec.get_cells()

            for (cell_id,cell) in rec.cells.items():

                self.cells |= {"%s_%s_%s"%(str(group_id),str(rec_id),str(cell_id)):cell}

        return self.cells

    def get_responsive(self):

        """
        Get a list containing the ids of all the responsive cells
        
        """

        ids = []

        for cell in self.cells:

            if self.cells[cell].responsive:

                ids.append(cell)

        return ids

    def get_populations(
        self,
        stims_names=None,
        trials_names=None,
        n_clusters=None,
        use_tsne=False,
        type="dff",
        normalize="norm",
        plot=True,
        ):

        """
        Clusterize the activity traces of all the cells into population using PCA and K-means.

        - stims_names: list
            The stimulation conditions to use for extracting the response feature of each cell.
            By default the respopnses to all the stimuli will be used.
        - trials_names: list
            The responses to the specified trials will be concatenated and used as features for
            th PCA, for all the stimuli specified by stims_names.
            By default the respopnses to all the trial types will be used.
        - n_clusters: int
            Number of cluster to use for k means clustering
        - use_tsne: bool
            wether to compute tsne embedding after PCA decomposition
        - type: str
            can be either "dff" or "zspks"
        - normalize: str
            'norm': signals will be normalized between 0 and 1 before running PCA
            'z': signals will be normalized using z score normalization before running PCA
            otherwise, no normalization will be applied

        """
        ### TO DO: SPLIT THIS FUNCTION IN:
        ### _compute fingerprints_(), PCA_embedding(), TSNE_embedding(), kmeans()


        # check if the cells have already been retrived
        if self.cells == None:

            self.get_cells()

        all_mean_resp = []

        if stims_names == None:

            stims_names = list(self.stims_trials_intersection.keys())

        responsive = self.get_responsive()

        for cell in responsive:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for stim in stims_names:

                # check if specified stims are in stims_trials_intersection
                if stim not in self.stims_trials_intersection:

                    warnings.warn("WARNING: stimulus '%s' is not shared by all the recordings,so itwill be skipped"%stim, 
                                  RuntimeWarning)
                    
                    break

                if trials_names == None:

                    trials_names = self.stims_trials_intersection[stim]

                for trial_name in trials_names:

                    # check if specified trial names are in stims_trials_intersection, for each stimuli
                    if trial_name not in self.stims_trials_intersection[stim]:

                        warnings.warn("WARNING: trial '%s' is not shared by stimulus '%s' all the recordings,so it will be skipped"%(stim,trial_name), 
                                      RuntimeWarning)

                    r = average_resp[stim][trial_name]["average_" + type]

                    # cut the responses
                    start = average_resp[stim][trial_name]['window'][0]
                    stop = average_resp[stim][trial_name]['window'][1]

                    r = r[start:int(stop+start/2)]
                    # low-pass filter 
                    r = filter(r,0.3)

                    concat_stims = np.concatenate((concat_stims, r))
                    
            if normalize == "norm":

                concat_stims = (concat_stims - concat_stims.min()) / (concat_stims.max() - concat_stims.min())

            elif normalize == "z":

                concat_stims = z_norm(concat_stims, True)

            all_mean_resp.append(concat_stims)

        # check lenghts consistency
        all_mean_resp = check_len_consistency(all_mean_resp)
        
        # convert to array
        # all_mean_resp = np.array(all_mean_resp)
        x = np.array(all_mean_resp)

        # run PCA and tSNE if desired

        if use_tsne:

            if len(x)<50:
                n_comp = len(x)
            else:
                n_comp = 50
                
            # run PCA
            pca = PCA(n_components=n_comp)
            transformed = pca.fit_transform(x)
            # run t-SNE
            tsne = TSNE(n_components=2, 
                        verbose=1, 
                        metric='cosine', 
                        early_exaggeration=4, 
                        perplexity=15, 
                        n_iter=2000, 
                        init='pca', 
                        angle=0.1)
            
            transformed = tsne.fit_transform(transformed)
        
        else:

            # run PCA
            pca = PCA(n_components=50)
            transformed = pca.fit_transform(x)

        # if the nuber of cluster is not specified, find optimal n
        if n_clusters == None:

            n_clusters = find_optimal_kmeans_k(transformed)

        # run Kmeans
        kmeans = KMeans(n_clusters=n_clusters, 
                        init="k-means++",
                        algorithm="auto").fit(transformed)
        
        labels = kmeans.labels_

        # retrive clusters
        clusters = []

        for n in np.unique(labels):

            indices = np.squeeze(np.argwhere(labels == n))
            c = []

            for i in indices:

                c.append(responsive[i])

            clusters.append(c)

        if plot:

            clist = list(colors.TABLEAU_COLORS.keys())
            markers = list(Line2D.markers.items())[2:] 
            # random.shuffle(markers)

            if use_tsne:

                algo = "t-SNE"

                Xax = transformed[:, 0]
                Yax = transformed[:, 1]

                fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111)

                fig.patch.set_facecolor("white")

                for l in np.unique(labels):

                    color = clist[l]
                    ix = np.where(labels == l)[0]

                    for i in ix:

                        marker = markers[int(responsive[i].split('_')[0])][0]
                        ax.scatter(
                            Xax[i], Yax[i], 
                            edgecolor=color,
                            s=50, 
                            marker=marker,
                            facecolors='none',
                            alpha=0.8
                        )

                ax.set_xlabel("%s 1"%algo, fontsize=9)
                ax.set_ylabel("%s 2"%algo, fontsize=9)

                ax.set_title("%d ROIs (n=%d)"%(len(Xax),len(self.recs)))

            else:

                algo = "PCA"

                Xax = transformed[:, 0]
                Yax = transformed[:, 1]
                Zax = transformed[:, 2]

                fig = plt.figure(figsize=(7, 5))
                # ax = fig.add_subplot(111, projection="3d")
                ax = fig.add_subplot(111)

                fig.patch.set_facecolor("white")

                for l in np.unique(labels):

                    color = clist[l]
                    ix = np.where(labels == l)[0]
                    # ax.scatter(
                    #     Xax[ix], Yax[ix], Zax[ix], c=clist[l], s=40)
                    
                    for i in ix:

                        marker = markers[int(responsive[i].split('_')[0])][0]
                        ax.scatter(
                            Xax[i], Yax[i], 
                            edgecolor=color,
                            s=50, 
                            marker=marker,
                            facecolors='none',
                            alpha=0.8
                        )

                ax.set_xlabel("%s 1"%algo, fontsize=9)
                ax.set_ylabel("%s 2"%algo, fontsize=9)
                # ax.set_zlabel("%s 3"%algo, fontsize=9)

                ax.set_title("%d ROIs (n=%d)"%(len(Xax),len(self.recs)))

                # ax.view_init(30, 60)


        return clusters


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
  
def filter(s, wn, ord=4, btype="low"):

    """
    Apply scipy's filtfilt to signal s.
    """

    b, a = butter(ord, wn, btype)
    s_filtered = filtfilt(b, a, s)

    return s_filtered

def z_norm(s, include_zeros=False):

    """
    Compute z-score normalization on signal s
    """

    if not isinstance(s, np.ndarray):

        s = np.array(s)

    if include_zeros:

        s_mean = np.mean(s)
        s_std = np.std(s)

        return (s - s_mean) / s_std

    elif s[s != 0].shape[0] > 1:

        s_mean = np.mean(s[s != 0])
        s_std = np.std(s[s != 0])

        return (s - s_mean) / s_std

    return np.zeros(s.shape)

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
    
def check_len_consistency(sequences):

    """
    Utility function for correcting for length inconsistency in a list of 1-D iterables.
    It finds the size of the shortes iterable and trim the other iterables accordingly.

    - sequence (list of iterables)
        list of array-like elements that will be trimmed to the same (minimal) length.

    """

    # find minimal length

    lengths = [len(sequence) for sequence in sequences]

    min_len = np.min(lengths)

    # trim to minimal length

    sequences_new = []

    for sequence in sequences:

        sequences_new.append(sequence[:min_len])

    return sequences_new

    