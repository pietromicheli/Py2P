import os
from pathlib import Path
import shutil
import argparse
import suite2p

parser = argparse.ArgumentParser(
                    prog='run_suite2P',
                    usage='run_suite2P data_path_1 ... data_path_n  -s [optional] save_path_1 ... save_path_n',
                    description='The program will run the basic suite2p pipeline for '
                                'all the recording files (tiff by default) found in '
                                'each of the specified data_path.b' 
                                'if the -s (save_paths) option is not specified, ' 
                                'the program will save the suite2p output in data_path. ' 
                                'Otherwise, you can use the -s option for specyfing the '
                                'locations where to save the output, one for each of the '
                                'data_paths that are passed.')
                   
parser.add_argument('data_paths', nargs='+', help='(list [str], input data paths)')    
parser.add_argument('-s', '--save_paths', nargs='+', help='(list [str], output save paths)')                    


def main(args):

    dataDirs = args.data_paths

    if args.save_paths:
        saveDirs = args.save_paths
    else:
        saveDirs = dataDirs

    if len(dataDirs) != len(saveDirs):

        print("> ERROR: You can't specify only one save_path if multiple data_path are provided!")
        return 0

    # set parameters
    ops = suite2p.default_ops()

    ops['input_format'] = "tif"   
    ops['nchannels'] = 1
    ops['tau'] = 0.11 #gCaMP8f
    ops['fs'] = 15.49
    ops['reg_tif'] = 0
    ops['do_registration'] = 1
    ops['denoise'] = 0
    ops['max_overlap'] = 0.5
    ops['anatomical_only'] = 3
    ops['diameter'] = 5
        
    for i,(dataDir,saveDir) in enumerate(zip(dataDirs,saveDirs)):

        # check if multiple recordings exist in the current data_path
        files =  [file for file in os.listdir(dataDir) 
                    if file.endswith(ops['input_format'])]

        rec_names = ['_'.join(Path(file).stem.split('_')[:3]) for file in files]

        if len(rec_names) > 1:

            # separate recording files in different directories
            new_dataDirs = []
            new_saveDirs = []

            for name in sorted(set(rec_names)):

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
    for dataDir,saveDir in zip(dataDirs,saveDirs):
        
        db = {
            'data_path': [dataDir],
            'save_path0': saveDir,
                }
        suite2p.run_s2p(ops=ops, db=db)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)