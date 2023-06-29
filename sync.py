import numpy as np
from scipy import io
import json


class Sync:

    """
    A Sync object that allows to allign a Record object to a Recording object.
    """

    def __init__(self):

        """
        Initialize synchroniuzer object
        """

        self.sync_frames = []
        self.stims_names = []
        self.trials_names = {}
        self.sync_ds = None

    def generate_data_structure(
        self,
        sync_file: str,
        stims_sequence_file: str,
        trials_names: dict,):

        """
        Create a sync data structure from scanbox metadata file.
        It requires also a .npy file where the full sequence of stimuli is stored.
        This class is ment to work with recordings where different types of stimuli
        were presented to either ipsilateral, controlateral or both eyes, and the frames
        corresponding to the onset and offset of each trial stored in a .mat file.

        - sync_file:
            .mat file containing a sequence of frames. Starting from the first element,
            every pair of frames will be considered as the onset and the offset of the trials (or events) specified
            in the stims_sequence_file.
        - stims_sequence_file:
            absolute path to .json file containing the full sequence of stimuli presented
            during the recording.
            The dict should contain pairs (stim_name:[trial_a1,...,trial_z1,...,trial_an,...,trial_zn])
            The different trial types (or events) should be encoded as a sequence of int values, fro 0 to (#trials types -1)
        - stim_names:
            list of strings containing the names of each stimuli type. It is assumed to to have a
            dimension equal to the first dimension of the data structure decribed by the stims_sequence_file,
            and to have the correct order.
        -trials_names:
            dict containing the names of the trials associated to the values contained in the stim_sequence_file,
            in the form: {value_0 : "trial_name_0", ... , value_n : "trial_name_n"}. All the in values contained
            in the stim_sequence_file must be associated to a trial name.

        """
        self.__init__()
        
        with open(stims_sequence_file, "r") as f:

            self.stims_sequence = json.load(f) 

        self.sync_frames = np.squeeze(io.loadmat(sync_file)["info"][0][0][0])
        self.stims_names = list(self.stims_sequence.keys())
        self.trials_names = trials_names

        """
        Generate a data structure where to store all the onset and offset frames
        for each trial type, for each stimulus type
        """

        sync_ds = {}

        # retrive all the (on_frame,off_frame) for all the trials
        i = 0
        for stim in self.stims_sequence:

            sync_ds |= {stim: {}}
            sequence = self.stims_sequence[stim]
            stim_start = self.sync_frames[i]

            for trial_type in np.unique(sequence):

                trial_len = self.sync_frames[i + 1] - self.sync_frames[i]
                pause_len = self.sync_frames[i + 2] - self.sync_frames[i + 1]
                sync_ds[stim] |= {
                    self.trials_names[trial_type]: {
                        "trials": [],
                        "trial_len": trial_len,
                        "pause_len": pause_len,
                    }
                }

            for trial in sequence:

                sync_ds[stim][self.trials_names[trial]]["trials"].append(
                    (self.sync_frames[i], self.sync_frames[i + 1])
                )

                i += 2

            stim_end = (self.sync_frames[i - 1]+
                        sync_ds[stim][self.trials_names[trial]]["pause_len"])

            sync_ds[stim] |= {"stim_window": (stim_start, stim_end)}

            # # make sure that all the trials have the same len across stimuli
            # for trial_type in self.trials_names.values():

            #     # trial len of first stimuli as reference
            #     trial_len = sync_ds[self.stims_names[0]][trial_type]["trial_len"]
            #     pause_len = sync_ds[self.stims_names[0]][trial_type]["pause_len"]

            #     for stim in self.stims_names:

            #         if sync_ds[stim][trial_type]["trial_len"] != trial_len:

            #             sync_ds[stim][trial_type]["trial_len"] = trial_len
                        
            #         if sync_ds[stim][trial_type]["pause_len"] != pause_len:

            #             sync_ds[stim][trial_type]["pause_len"] = pause_len

        
        self.sync_ds = sync_ds

        return self

    def load_data_structure(
            self, 
            ds_json):

        """
        Load data structure from a json file.
        See the example sync_dict_example.json for the structure that the input ds_json file must have.
        """
        self.__init__()

        with open(ds_json, "r") as f:

            self.sync_ds = json.load(f)


        for stim in self.sync_ds:

            self.stims_names.append(stim)

            for i,trial_type in enumerate(self.sync_ds[stim]):
            
                if trial_type not in self.trials_names.values():

                    self.trials_names |= {i:trial_type}

                for trial in self.sync_ds[stim][trial_type]["trials"]:

                    self.sync_frames.extend(trial)

        return self
                        
    def generate_data_structure_sn(
        self,
        sync_file: str,
        texture_file: str,
        trial_len=20,
        ):

        """
        Crete a sync data structure specifically for Sparse Noise recordings.
        It requires the scanbox metadata .mat file and the sparse noise texture .mat file
        
        - sync_file:
            .mat file containing a sequence of frames. Starting from the first element,
            every pair of frames will be considered as the onset and the offset of the trials (or events) specified
            in the stims_sequence_file.
        - texture_file:
            .mat file containing the sequence of matrices which represent the textures presented during the sparse noise
            stimulation.
        - trial_len: int
            number of frames that will be considered as part of a trial, starting from the onset frame present in sync_file
        """     

        self.sync_frames = np.squeeze(io.loadmat(sync_file)["info"][0][0][0])
        self.textures = io.loadmat(texture_file)['stimulus_texture']
        self.text_dim = self.textures.shape
        self.trial_len = trial_len
        self.stims_names.append("sparse_noise")
        self.sync_ds = {'sparse_noise':{}}

        for i in range(self.text_dim[1]):
            for j in range(self.text_dim[2]):

                on_indexes = np.where(self.textures[:,i,j]==1)[0]
                off_indexes = np.where(self.textures[:,i,j]==0)[0]

                # extract sync frames where square turned white
                on_frames = [(frame,frame+trial_len) for frame in self.sync_frames[on_indexes]]
                # extract sync frames where square turned black
                off_frames = [(frame,frame+trial_len) for frame in self.sync_frames[off_indexes]]

                # on trial
                # self.trials_names.append('%d_%d_on'%(i,j))
                self.sync_ds['sparse_noise']|=({'%d_%d_white'%(i,j): 
                                                {'trials': on_frames,
                                                'trial_len': trial_len,
                                                'pause_len': 0}})
                # off trial 
                # self.trials_names.append('%d_%d_off'%(i,j))
                self.sync_ds['sparse_noise']|=({'%d_%d_black'%(i,j): 
                                                {'trials': off_frames,
                                                'trial_len': trial_len,
                                                'pause_len': 0}})

        self.sync_ds['sparse_noise']|={'stim_window':(self.sync_frames[0],self.sync_frames[-1])}

        return self  