import numpy as np
from scipy import io 
import os
import sys

def merge_mat_files(mat_files, offsets):

    """
    merge sync mat file, adding to each value the offset specified.

    -mat_files (list of str):
    list of .mat sync filenames to concatenate

    -offsets (list of int)
    list of offset values to add to each file. N.B: offset length should be equal to mat_files length

    """

    path = os.path.split(mat_files[0])[0]

    sync_conc = None

    for filename, offset in zip(mat_files, offsets):

        sync = np.int64(np.squeeze(io.loadmat(filename)["info"][0][0][0]))

        if sync_conc is None:

            sync_conc = sync

        else:
            sync = sync + offset
            sync_conc = np.concatenate((sync_conc, sync), axis=None)

    mat_dict = {"info": np.array([np.array([np.array([sync_conc])])])}
    io.savemat(
        path + "\\sync_merged.mat",
        mat_dict,
    )

    print("\n> sync file merged and saved in %s"%path)

    return path + "sync_merged.mat"


if __name__ == "__main__":

    import argparse
    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--mat_files",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[],  # default if nothing is provided
    )
    CLI.add_argument(
    "--offsets",
    nargs="*",
    type=int, 
    default=[],
    )

    # parse the command line
    args = CLI.parse_args()

    merge_mat_files(args.mat_files,args.offsets)