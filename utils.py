import os
import shutil
from glob import glob

def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def rmdir_if_existed(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)

def rmfile_if_existed(file_name):
    fpaths = glob(file_name)
    for fpath in fpaths:
        if os.path.exists(fpath):
            os.remove(fpath)