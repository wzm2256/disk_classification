import random
import json
import PIL.Image as Image
import os
import numpy as np
import pdb
import torch
from torchvision import transforms

def Extract_json(f):
    json1_data = json.load(f)
    Dict = {}
    for s in json1_data['shapes']:
        Dict[s["label"]] = s["points"]
    return Dict

def crop_I(I, points):
    p1 = points[0]
    p2 = points[1]

    area = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
    I_crop = I.crop(area)
    return I_crop

def collect_positive_label(folder):
    Pos_Dic = {}
    f = open(os.path.join(folder, 'labels.txt'))
    f.readline()
    for line in f.readlines():
        if line.strip():
            tmp = line.strip().split(' ')
            Pos_Dic[tmp[0]] = []
            for i in tmp[1:]:
                i_strip = i.strip()
                if i_strip:
                    Pos_Dic[tmp[0]].append(i.strip())
    return Pos_Dic

def pad_resize(image, resize):
    I = image.convert('RGB')
    I = transforms.ToTensor()(I)
    h = I.shape[1]
    w = I.shape[2]
    w_pad = (1024 - w) // 2
    h_pad = (1024 - h) // 2
    assert w_pad > 0, 'Width too large'
    assert h_pad > 0, 'Height too large'
    Large_tensor = torch.nn.functional.pad(I, (w_pad, w_pad, h_pad, h_pad), mode='constant')
    Large_image =transforms.ToPILImage()(Large_tensor)
    image_new = Large_image.resize(resize)
    return image_new


def Collect_positive_sample(Input_folder, Output_folder, Positive_Dict, D, num=None, resize=(256, 256)):
    f = open(os.path.join(Input_folder, 'labels.txt'))
    foldertype = f.readline().strip()
    f.close()

    All_list = []
    for i in Positive_Dict.keys():
        for j in Positive_Dict[i]:
            All_list.append('{}_{}_{}'.format(i, foldertype, j.zfill(2)))

    if num is None:
        Select_list = All_list
    elif num >= len(All_list):
        # pdb.set_trace()
        Select_list1 = np.random.choice(All_list, size=num-len(All_list))
        Select_list = All_list + list(Select_list1)
    elif num < len(All_list):
        Select_list = np.random.choice(All_list, size=num)

    count = 0
    for i in sorted(Select_list):
        with Image.open(os.path.join(Input_folder, i.split('_')[0]+'.jpg')) as I:
            label = i.split('_')[1] + '_' + i.split('_')[-1]
            cor = D[label]
            image = crop_I(I, cor)
            image = pad_resize(image, resize)
            # print(image.size)
            # if resize is not None:
            #     image = image.resize(resize)
            # pdb.set_trace()
            # image.save(os.path.join(Output_folder, '{}_{}_1.png'.format(count, foldertype)))
            image.save(os.path.join(Output_folder, '{}_{}_1.png'.format(count, label)))
            count += 1
    return count

def Collect_negative_sample(Select_folder, Output_folder, Positive_Dict, num, D, resize=(256, 256), All_folder=None):
    if All_folder is None:
        All_folder = ['Disks/1', 'Disks/3', 'Disks/2']
    Others = [i for i in All_folder if i != Select_folder]
    # pdb.set_trace()

    f = open(os.path.join(Select_folder, 'labels.txt'))
    foldertype = f.readline().strip()
    f.close()

    Label_set = set()
    for i in Positive_Dict.keys():
        # Label_set += set(Positive_Dict[i])
        Label_set = Label_set.union(Positive_Dict[i])

    Negative_list = []
    for i in Positive_Dict.keys():
        for j in list(Label_set - set(Positive_Dict[i])):
            Negative_list.append('{}_{}_{}'.format(i, foldertype, j.zfill(2)))

    size = int(0.8 * num // len(D))
    count = 0


    for l in Others:
        All = [i for i in os.listdir(l) if i.endswith('.jpg')]
        Select_list = np.random.choice(All, size=size // 2+1)
        for i in Select_list:
            for j in D.keys():
                with Image.open(os.path.join(l, i)) as I:
                    I_crop = crop_I(I, D[j])
                    # print(image.size)
                    # pdb.set_trace()
                    if resize is not None:
                        # I_crop = I_crop.resize(resize)
                        I_crop = pad_resize(I_crop, resize)
                    # I_crop.save(os.path.join(Output_folder, '{}_{}_0.png'.format(count, foldertype)))
                    I_crop.save(os.path.join(Output_folder, '{}_{}_0.png'.format(count, j)))
                    count += 1
    
    size_same = int(0.2 * num)

    if size_same >= len(Negative_list):
        Select_list1 = np.random.choice(Negative_list, size=size_same-len(Negative_list))
        Select_list = Negative_list + list(Select_list1)
    elif size_same < len(Negative_list):
        Select_list = np.random.choice(Negative_list, size=size_same)


    for i in sorted(Select_list):
        with Image.open(os.path.join(Select_folder, i.split('_')[0]+'.jpg')) as I:
            label = i.split('_')[1] + '_' + i.split('_')[-1]
            cor = D[label]
            image = crop_I(I, cor)
            if resize is not None:
                # image = image.resize(resize)
                image = pad_resize(image, resize)
            # pdb.set_trace()
            # image.save(os.path.join(Output_folder, '{}_{}_1.png'.format(count, foldertype)))
            image.save(os.path.join(Output_folder, '{}_{}_0.png'.format(count, label)))
            count += 1

    return count


def Collect_for_test(file_name, Output_folder, D_list, resize=(256, 256)):
    with Image.open(file_name) as I:
        for D in D_list:
            for j in D.keys():
                I_crop = crop_I(I, D[j])
                # I_crop = I_crop.resize(resize)
                I_crop = pad_resize(I_crop, resize)
                # I_crop.save(os.path.join(Output_folder, '{}_{}_0.png'.format(count, foldertype)))
                I_crop.save(os.path.join(Output_folder, 'Test_{}_9.png'.format(j)))


def split_dataset(split):
    All_name = [i for i in os.listdir('./Crop/') if i.endswith('png')]
    random.shuffle(All_name)
    # pdb.set_trace()
    Train_list = All_name[:int(len(All_name) * split)]
    Test_list = All_name[int(len(All_name) * split):]
    f = open('Crop/train.txt', 'w')
    for i in Train_list:
        f.write(i + '\n')
    f.close()

    f = open('Crop/test.txt', 'w')
    for i in Test_list:
        f.write(i + '\n')
    f.close()
