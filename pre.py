import shutil
import pathlib
import os
import numpy as np


def main():
    """
    """
    # Get all the files in the current directory
    files = pathlib.Path('./').glob('data/Int_val/*/*.png')
    for f in files:
        # print(f.split('/')[-1])
        f = str(f)
        if f.split('/')[-1].startswith('.'):
            os.remove(f)#, os.path.join('data/Int_val', f.split('/')[-2], f.split('/')[-1][1:]))


def make_5fold():
    AD_files = pathlib.Path('./').glob('data/Train/AD/*.png')
    HP_files = pathlib.Path('./').glob('data/Train/HP/*.png')
    if not os.path.exists('data/fold'):
        os.mkdir('data/fold')
        for i in range(1, 6):
            os.makedirs(f'data/fold/{i}/Train/AD')
            os.makedirs(f'data/fold/{i}/Train/HP')
            os.makedirs(f'data/fold/{i}/Int_val/AD')
            os.makedirs(f'data/fold/{i}/Int_val/HP')
    
    AD_files = list(AD_files)
    HP_files = list(HP_files)
    # Determine the correct indices to split the data.
    AD_limits = np.linspace(0, len(AD_files)+1, 6, dtype=int)
    HP_limits = np.linspace(0, len(HP_files)+1, 6, dtype=int)

    for i in range(len(AD_limits) - 1):
        # Split the data at the correct indices and save it.
        AD_files_split_val = AD_files[AD_limits[i]:AD_limits[i+1]]
        HP_files_split_val = HP_files[HP_limits[i]:HP_limits[i+1]]
        
        AD_files_split_train = AD_files[:AD_limits[i]] + AD_files[AD_limits[i+1]:]
        HP_files_split_train = HP_files[:HP_limits[i]] + HP_files[HP_limits[i+1]:]
        for f in AD_files_split_val:
            f = str(f)
            shutil.copy(f, f'data/fold/{i+1}/Int_val/AD/' + f.split('/')[-1])
        for f in HP_files_split_val:
            f = str(f)
            shutil.copy(f, f'data/fold/{i+1}/Int_val/HP/' + f.split('/')[-1])
            # print(f'data/fold/{i+1}/val/HP' + f.split('/')[-1])
        for f in AD_files_split_train:
            f = str(f)
            shutil.copy(f, f'data/fold/{i+1}/Train/AD/' + f.split('/')[-1])
        for f in HP_files_split_train:
            f = str(f)
            shutil.copy(f, f'data/fold/{i+1}/Train/HP/' + f.split('/')[-1])


if __name__ == '__main__':
    make_5fold()
