import numpy as np
import joblib
import os

pathbox = '/media/tonyw/SSD/Workspace/JupyterNotebook/testset_new/test_new/bbox/'

for fpathe, dirs, fs in os.walk(pathbox):
    time_stamps = []
    for f in fs:
        print(f)
        time_stamp = f.split('.')[0] + '.' + f.split('.')[1]
        time_stamps.append(time_stamp)
    joblib.dump(time_stamps, '/media/tonyw/SSD/Workspace/JupyterNotebook/testset/time_stamps_tmp')
