import test
import os
import subprocess
import pack_test
from tqdm import tqdm
# All_folders = ['Disks/1', 'Disks/2', 'Disks/3']
All_folders = ['Disks/2', 'Disks/3']

# All_folders = ['Disks/2']
# for model_name in ['pack/best0.pt', 'pack/best1.pt', 'pack/best2.pt', 'pack/best3.pt', 'pack/best4.pt']:
# for model_name in ['pack/best_0507.pt']:
for model_name in ['pack/new_best_1210.pt']:
    print(model_name)
    for f in All_folders:
        file = os.path.join(f, 'labels.txt')
        labels = open(file)
        c = labels.readline().strip()

        for line in tqdm(labels.readlines()):
            if not line.strip():
                continue
            tmp_list = line.strip().split(' ')
            image_name = tmp_list[0] + '.jpg'
            if len(tmp_list) == 1:
                labels = []
            else:
                labels = ['{}_{}'.format(c, i.zfill(2)) for i in line.strip().split(' ')[1:]]
            image_path = os.path.join(f, image_name)
            # Predict = test.test_image(image_path)
            Predict = pack_test.test_image(image_path, model_name=model_name)
            
            if Predict != labels:
                print(image_path)
                print(labels)
                print(Predict)
            else:
                pass
                # print(image_path + ': Pass')
            # print(image_path + '*******:')
            # print(labels)
            # subprocess.run('python test.py {}'.format(image_path), shell=True)

