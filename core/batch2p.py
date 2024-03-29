import numpy as np

from .rec2p import R2p 
from Py2P.utils import *
from Py2P.plot import plot_clusters

class B2p:

    """
    Class for performing batch analysis of multiple recordings.
    All the recordings contained in a Batch2P object must share at
    least one stimulation condition with at least one common trial
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
            By default, all recordings loaded are assigned to group 0.
        """

        # be sure that the sync object of all the recordings share at least 1 stimulus.
        # only shared stimuli will be used.

        self.stims_trials_intersection = {}

        stims_allrec = [sync.stims_names for sync in data_dict.values()]

        # start from minimal stim set
        stims_intersection = set(stims_allrec[np.argmin([len(s) for s in stims_allrec])])

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
                trials_intersection.intersection(
                    set(list(sync.sync_ds[stim].keys())[:-1])
                )

            self.stims_trials_intersection |= {stim: list(trials_intersection)}

        # generate params.yaml
        generate_params_file()

        # instantiate the Rec2P objects
        self.recs = {}
        self.groups = groups

        for rec_id, (data_path, sync) in enumerate(data_dict.items()):

            if data_path not in groups:
                group_id = 0

            else:
                group_id = groups[data_path]

            rec = R2p(data_path, sync)
            self.recs |= {rec_id: (rec, group_id)}

        self.cells = None
        self.populations = []

    def load_params(self):

        """
        Read parameters from .yaml params file

        """

        for rec in self.recs:

            self.recs[rec][0].load_params()

    def get_cells(self):

        """
        Extract all the cells from the individual recordings and assign new ids.
        Id is in the form G_R_C, where G,R and C are int which specify the group,
        the recording and the cell ids.

        """

        self.cells = {}

        for (rec_id, value) in self.recs.items():

            rec = value[0]
            group_id = value[1]

            # retrive cells for each recording
            rec.get_cells()

            for (cell_id, cell) in rec.cells.items():

                # new id
                new_id = "%s_%s_%s" % (str(group_id), str(rec_id), str(cell_id))
                self.cells |= {new_id: cell}
                # update cell id
                cell.id = new_id

        # RETRIVE THE GROUPS
        self.cells_groups = {g:{} for g in set(self.groups.values())}

        for id,cell in self.cells.items():
            for g in self.cells_groups:

                if int(id.split('_')[0])==g:

                    self.cells_groups[g] |= {id:cell}

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

    def compute_fingerprints(
        self, 
        cells_ids=None,
        stim_trials_dict=None, 
        type="dff", 
        normalize="z", 
        smooth=True
    ):

        """
        Compute a fingerprint for each cell by concatenating the average responses
        to the specified stimuli and trials.

        - cells_ids: list of valid ids
            by default, compute fingerprints of all the responsive cells
        - stim_trials_dict: dict
            A dict which specifies which stim and which trials to concatenate for computing
            the fingerptint.
            Should contain key-values pairs such as {stim:[t1,...,tn]}, where stim is a valid
            stim name and [t1,...,tn] is a list of valid trials for that stim.
        """

        # check if the cells have already been retrived

        if self.cells == None:

            self.get_cells()

        if stim_trials_dict == None:

            stim_trials_dict = {stim: [] for stim in self.stims_trials_intersection}

        responsive = self.get_responsive()

        fingerprints = []

        if cells_ids == None:
            
            cells_ids = responsive

        for cell in cells_ids:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for (stim, trials_names) in stim_trials_dict.items():

                if not trials_names:

                    trials_names = list(self.stims_trials_intersection[stim])

                for trial_name in trials_names:

                    r = average_resp[stim][trial_name]["average_%s" % type]

                    # cut the responses
                    start = average_resp[stim][trial_name]["window"][0]
                    stop = average_resp[stim][trial_name]["window"][1]

                    r = r[start : int(stop + start / 2)]

                    if smooth:
                        # low-pass filter
                        r = filter(r, 0.3)

                    concat_stims = np.concatenate((concat_stims, r))

            if normalize == "lin":

                concat_stims = lin_norm(concat_stims, -1, 1)

            elif normalize == "z":

                concat_stims = z_norm(concat_stims, True)

            fingerprints.append(concat_stims)

        # check lenghts consistency
        fingerprints = check_len_consistency(fingerprints)

        # convert to array
        fingerprints = np.array(fingerprints)

        ## NB: index consistency between fingerprints array and list from get_responsive() is important here!

        return fingerprints

    def get_populations(
        self,
        cells_ids=None,
        algo='pca',
        clusters='kmeans',
        k=None,
        markers=True,
        save_name=None,
        groups_name=None,
        tsne_params=None,
        marker_mode=1,
        **kwargs
        ):

        '''

        Find functional populations within the set of cells specified. 
        
        - cells_id: list of str
            list of valid cells ids used for identify which subset of all the cells to analyze.
            By thefault, all the cells present in the batch will be analyzed.
        - algo: str
            algorithm for demensionality reduction. Can be pca or tsne.
        - clusters: str
            clustering algorithm. can be "kmeans" or "gmm" (gaussian mixture model)
        - k: int
            number of expected clusters.
        - marker_mode: int
            0: markers represent the groups
            1: markers represent the recordings
        - **kwargs:
            any valid argument to parametrize compute_fingerprints() method

        '''

        fp = self.compute_fingerprints(
                    cells_ids = cells_ids,
                    **kwargs)
        
        if algo=='pca':
            
            # run PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(fp)
            exp_var = pca.explained_variance_ratio_
            xlabel = "PC1 ({} %)".format(round(exp_var[0]*100,2))
            ylabel = "PC2 ({} %)".format(round(exp_var[1]*100,2))

        elif algo=='tsne':

            # if needed, go with Tsne
            if tsne_params==None:
                tsne_params =  {
                        'n_components':2, 
                        'verbose':1, 
                        'metric':'cosine', 
                        'early_exaggeration':4, 
                        'perplexity':10, 
                        'n_iter':3000, 
                        'init':'pca', 
                        'angle':0.9}

            transformed = TSNE_embedding(fp,**tsne_params)
            xlabel = "Dimension 1"
            ylabel = "Dimension 2"

        # clusterize
        # labels = GMM(transformed,n_components=n_components,covariance_type='diag')
        if k == None:
            k = find_optimal_kmeans_k(transformed)

        if clusters == 'kmeans':
            labels = k_means(transformed, k)
        elif clusters == 'gmm':
            labels = GMM(transformed,n_components=(k),covariance_type='diag')

        if markers:
            markers = [int(id.split(sep='_')[marker_mode]) for id in cells_ids]

        else:
            markers=None

        plot_clusters(transformed,labels,markers,xlabel,ylabel,groups_name=groups_name,save=save_name)

        # get popos
        pops = []
        for n in np.unique(labels):

            indices = np.where(labels == n)[0]

            c = []
            for i in indices:

                c.append(cells_ids[i])

            pops.append(c)

        self.populations.append(pops)

        # assign pop label at each cell
        for i,pop in enumerate(pops):
            for id in pop:

                self.cells[id].label = i

        return pops

