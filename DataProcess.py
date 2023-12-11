import os
import json
import PIL.Image as Image
import util as util
import numpy as np
import pdb
import random
import argparse

agsps = argparse.ArgumentParser()
agsps.add_argument('--DiskPath', default='Disks')
agsps.add_argument('--TemplatePath', default='template_bbx')
args = agsps.parse_args()


All_folders = [os.path.join(args.DiskPath, '1'), os.path.join(args.DiskPath, '2'), os.path.join(args.DiskPath, '3')]

D = []
for i in range(3):
    f = open( os.path.join(args.TemplatePath, f'{i+1}.json'))
    D.append(util.Extract_json(f))
    f.close()

# pdb.set_trace()
# Collect Samples
Pos = [util.collect_positive_label(All_folders[i]) for i in range(3)]

# pdb.set_trace()
Pos_Num = [  np.sum(np.array([len(i[j])  for j in i.keys()])) for i in Pos]
print('Numbers of positive samples in each class are {}, {}, {}'.format(Pos_Num[0], Pos_Num[1], Pos_Num[2]))
Select_positive_number = int(np.median(Pos_Num))
# Select_positive_number = int(np.max(Pos_Num) * 1.5)
# Select_positive_number = int(np.median(Pos_Num) * 0.005)
print('Select Samples:{}'.format(Select_positive_number) )

print('Preparing Crop folder')
if os.path.isdir('Crop'):
    All = os.listdir('Crop')
    for i in All:
        os.remove(os.path.join('Crop', i))
print('Done')

for i in range(3):
    print(f'Processing disk {i}...')
    n_p = util.Collect_positive_sample(All_folders[i], './Crop/', Pos[i], D[i], Select_positive_number)
    print('Generating {} positive samples for dataset {}'.format(n_p, i))
    n_n = util.Collect_negative_sample(All_folders[i], './Crop/', Pos[i], Select_positive_number, D[i], All_folder=All_folders)
    print('Generating {} negative samples for dataset {}'.format(n_n, i))

# Random split

split = 0.8
util.split_dataset(split)