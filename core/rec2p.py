import numpy as np
import yaml
import warnings
from tqdm import tqdm
from sklearn.decomposition import PCA

from .sync import Sync
from .cell2p import Cell2p
from Py2P.utils import *
from Py2P.plot import plot_clusters

class Rec2p:

    """
    A recording master class
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
        self.Fraw = np.load(data_path + r"/F.npy")
        self.Fneu = np.load(data_path + r"/Fneu.npy")
        self.iscell = np.load(data_path + r"/iscell.npy")
        self.spks = np.load(data_path + r"/spks.npy")
        self.stat = np.load(data_path + r"/stat.npy", allow_pickle=True)
        self.ops = np.load(data_path + r"/ops.npy", allow_pickle=True)

        self.nframes = self.Fraw.shape[1]
        self.ncells = self.Fraw.shape[0]

        print("OK")

        # check if the end of the last stim_window specifyed by the sync structure exceeds 
        # the length of the recording. If not, pad the recording. This can be due to premature 
        # end of the recording, where the pause after the last trial is too short.

        if sync.sync_ds[sync.stims_names[-1]]['stim_window'][1]>self.nframes:
# 
            pad_len = sync.sync_ds[sync.stims_names[-1]]['stim_window'][1]-self.nframes

            self.Fraw = np.pad(self.Fraw,((0,0),(0,pad_len)),mode='constant',constant_values=((0,0),(0,np.mean(self.Fraw[-10:]))))
            self.Fneu = np.pad(self.Fneu,((0,0),(0,pad_len)),mode='constant',constant_values=((0,0),(0,np.mean(self.Fneu[-10:]))))
            self.spks = np.pad(self.spks,((0,0),(0,pad_len)),mode='constant',constant_values=((0,0),(0,np.mean(self.spks[-10:]))))

            print('> WARNING: last trial has been padded (pad length:%d, Fraw value:%d)'%(pad_len,np.mean(self.Fraw[-1])))
# 
        self.data_path = data_path
        self.params_file = generate_params_file()

        self.sync = sync
        self.cells = None
        self.params = None

        # read parameters from .yaml params file
        self.load_params()

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
        keep_unresponsive: bool=False, 
        n: int = None, 
        clean_memory=False
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

            idxs_cells = np.arange(0, self.ncells)

            if self.params["use_iscell"]:

                idxs_cells = np.where(self.iscell[:, 0] == 1)[0]
        else:

            idxs_cells = np.arange(0, n)

        for id in tqdm(idxs_cells):

            if self.params["use_iscell"]:

                # check the ROIS has been classified as cell by suite2p
                if not self.iscell[id][0]:
                    continue

            cell = Cell2p(self, id)
            cell.analyze()

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
                "> %d responsive cells found (tot: %d, keep_unresponsive: %r, use_iscell: %r)"
                % (
                    len(responsive),
                    self.ncells,
                    keep_unresponsive,
                    self.params["use_iscell"],
                )
            )

        if clean_memory:

            del self.Fraw
            del self.Fneu
            del self.spks
            del self.iscell

        return self.cells

    def compute_fingerprints(
        self, 
        cells_ids,
        stim_trials_dict=None, 
        type="dff", 
        normalize="z", 
        smooth=True
    ):

        """
        Compute a fingerprint for each cell by concatenating the average responses
        to the specified stimuli and trials.

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

            stim_trials_dict = {stim: [] for stim in self.sync.stims_names}

        responsive = self.get_responsive()

        fingerprints = []

        for cell in cells_ids:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for (stim, trials_names) in stim_trials_dict.items():

                if not trials_names:

                    trials_names = list(self.sync.stims_dict[stim])

                for trial_name in trials_names:

                    r = average_resp[stim][trial_name]["average_%s" % type]

                    # cut the responses
                    start = average_resp[stim][trial_name]["window"][0]
                    stop = average_resp[stim][trial_name]["window"][1]

                    r = r[start : int(stop + start / 2)]

                    if smooth:
                        # low-pass filter
                        r = filter(r, 0.1)

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
        n_components=2,
        save_name='',
        **kwargs
        ):

        '''

        Find functional populations within the set of cells specified. 
        
        - cells_id: list of str
            list of valid cells ids used for identifying which subset of all the cells to analyze.
            By thefault, all the cells present in the recording will be analyzed.
        - algo: str
            algorithm for demensionality reduction. Can be pca or tsne.
        - n_components: int
            number of component used by GMM for clustering.
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
            xlabel = "PC1 (% {})".format(round(exp_var[0],2))
            ylabel = "PC2 (% {})".format(round(exp_var[1],2))

        elif algo=='tsne':

            # if needed, go with Tsne
            tsne_params =  {
                    'n_components':2, 
                    'verbose':1, 
                    'metric':'cosine', 
                    'early_exaggeration':4, 
                    'perplexity':10, 
                    'n_iter':3000, 
                    'init':'pca', 
                    'angle':0.1}

            transformed = TSNE_embedding(fp,**tsne_params)
            xlabel = "Dimension 1"
            ylabel = "Dimension 2"

        # clusterize
        labels = GMM(transformed,n_components=n_components,covariance_type='diag')

        if save_name:
            plot_clusters(transformed,labels,xlabel=xlabel,ylabel=ylabel,save='%s_%s'%(save_name,algo))

        else:
            plot_clusters(transformed,labels,xlabel=xlabel,ylabel=ylabel,save='')

        # get popos
        pops = []
        for n in np.unique(labels):

            indices = np.where(labels == n)[0]

            c = []
            for i in indices:

                c.append(cells_ids[i])

            pops.append(c)

        return pops
    
 