import os
from pathlib import Path
import shutil
import argparse
from tifffile import TiffFile
from natsort import natsorted
from merge_sync_mat import merge_mat_files
import suite2p

parser = argparse.ArgumentParser(
                    prog='run_suite2P',
                    usage='run_suite2P data_path_1 ... data_path_n  -s [optional] save_path_1 ... save_path_n -m [optional]',
                    description='The program will run the basic suite2p pipeline for '
                                'all the recording files (tiff by default) found in '
                                'each of the specified data_path.' 
                                'if the -s (save_paths) option is not specified, ' 
                                'the program will save the suite2p output in data_path. ' 
                                'Otherwise, you can use the -s option for specyfing the '
                                'locations where to save the output, one for each of the '
                                'data_paths that are passed.')
                   
parser.add_argument('data_paths', nargs='+', help='(list [str], input data paths)')    
parser.add_argument('-s', '--save_paths', nargs='+', help='(list [str], output save paths)')                    
parser.add_argument('-m', '--merge', action='store_true', help='merge all the recordings found in the same data_path')

def main(args):

    dataDirs =  args.data_paths
    if args.save_paths:
        saveDirs = args.save_paths
    else:
        saveDirs = dataDirs

    if len(dataDirs) != len(saveDirs):
        print()

        print("> ERROR: The number of save paths should match the number of data paths!")
        return 0

    merge = args.merge

    # set parameters
    ops = suite2p.default_ops()

    ops['input_format'] = "tif"
    ops['nchannels'] = 1
    ops['tau'] = 0.11 #gCaMP8f
    ops['fs'] = 15.49
    ops['reg_tif'] = 0
    ops['do_registration'] = 1
    ops['delete_bin'] = 0
    ops['denoise'] = 0
    ops['max_overlap'] = 0.5
    ops['anatomical_only'] = 3
    ops['diameter'] = 5

    for i,(dataDir,saveDir) in enumerate(zip(dataDirs,saveDirs)):

        print('\n> DATA PATH: {}'.format(dataDir))

        # check if multiple recordings exist in the current data_path
        files =  natsorted([file for file in os.listdir(dataDir) 
                    if file.endswith(ops['input_format'])])

        rec_names = ['_'.join(Path(file).stem.split('_')[:3]) for file in files]
        print('> Rec Files:')
        for rname in rec_names: print(rname)

        if merge:

            # check if there are .mat files to merge
            mat_names = natsorted([name+'.mat' for name in rec_names if name+'.mat' in os.listdir(dataDir)])

            if len(mat_names) == len(rec_names):

                print('> found .mat sync files. Merging:')
                for mname in mat_names: print(mname)

                # join the mat files
                offset = 0
                offsets = [offset]
                for file in files[:-1]:

                    with TiffFile(os.path.join(dataDir,file)) as tif:
                        # Get the number of pages, which corresponds to the number of frames
                        n_frames = len(tif.pages)
                        offset += n_frames
                        offsets.append(offset)
                
                # merge mat files
                merge_mat_files([os.path.join(dataDir,name) for name in mat_names],offsets)

        elif len(rec_names) > 1:

            # separate recording files in different directories
            new_dataDirs = []
            new_saveDirs = []

            for name in rec_names:

                new_dataDir = os.path.join(dataDir,name)
                os.mkdir(new_dataDir)
                new_dataDirs.append(new_dataDir)

                if dataDir != saveDir:

                    new_saveDir = os.path.join(saveDir,name)
                    os.mkdir(new_saveDir)
                    new_saveDirs.append(new_saveDir)

                else:
                    new_saveDirs.append(new_dataDir)

                for file in files:

                    if name in file:

                        shutil.move(os.path.join(dataDir,file),new_dataDir)

                
            dataDirs.pop(i)
            dataDirs[i:i] = new_dataDirs
            saveDirs.pop(i)
            saveDirs[i:i] = new_saveDirs

        # run suite2p for all the recordings found
        print("\n> Running Suite2p ...")

        db = {
            'data_path': [dataDir],
            'save_path0': saveDir,
                }
        suite2p.run_s2p(ops=ops, db=db)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)