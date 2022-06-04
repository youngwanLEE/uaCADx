import shutil
import pathlib
import os


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


if __name__ == '__main__':
    main()
