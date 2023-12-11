import pdb
import torch
from torchvision import transforms
import model_util as model_util
import argparse
import util as util
import os
import numpy as np
from torch import package
import json

def test_image(file, model_name="pack/best0.pt", size=227):
    #### Preparing dataset
    D = []
    for i in range(3):
        f = open('template_bbx/{}.json'.format(i+1))
        D.append(util.Extract_json(f))
        f.close()

    with open("Dict.json", "w") as write_file:
        json.dump(D, write_file, indent=4)

    util.Collect_for_test(file, 'TMP', D)

    All_name = [i for i in os.listdir('./TMP/') if i.endswith('png')]
    f = open('TMP/test.txt', 'w')
    for i in All_name:
        f.write(i + '\n')
    f.close()

    ### Preparing dataset
    trans_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[1])
    ])

    D_test = model_util.My_dataset('TMP', 'test.txt', transform=trans_test)

    ## Load network
    path = model_name
    package_name = "Disk_model"
    resource_name = "model.pkl"
    # pdb.set_trace()
    imp = package.PackageImporter(path)
    my_network = imp.load_pickle(package_name, resource_name)

    #####
    full_position = []
    I = []
    for i, _, full in D_test:
        logit_i = my_network(i.unsqueeze(0))
        full_position.append(full)
        I.append(logit_i)
    logit = torch.cat(I, 0)
    #####
    _, predict = torch.max(logit, 1)

    Pre = []
    Pre_c = []
    for i in range(36):
        if predict[i].numpy() == 1:
            Pre.append(full_position[i])
            Pre_c.append(full_position[i].strip('_')[0])

    Count = []
    for i in range(3):
        Count.append(Pre_c.count(str(i+1)))

    Largest = np.argmax(np.array(Count))
    Select_label = [i for i in Pre if i.startswith(str(Largest+1))]

    return Select_label

if __name__ == '__main__':
    agsps = argparse.ArgumentParser()
    agsps.add_argument('file', type=str)
    agsps.add_argument('--size', type=int, default=227)


    args = agsps.parse_args()

    Select_label = test_image(args.file, args.size)
    print(Select_label)



