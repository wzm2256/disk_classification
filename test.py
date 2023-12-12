import pdb
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import model_util
import argparse
import util as util
import net1
import os
import numpy as np
import time

def test_image(args):
    # time1 = time.time()
    #### Preparing dataset
    D = []
    for i in range(3):
        f = open(f'{args.TemplatePath}/{i+1}.json')
        D.append(util.Extract_json(f))
        f.close()

    util.Collect_for_test(args.file, 'TMP', D)

    All_name = [i for i in os.listdir('./TMP/') if i.endswith('png')]
    f = open('TMP/test.txt', 'w')
    for i in All_name:
        f.write(i + '\n')
    f.close()

    # time2 = time.time()
    ### Preparing loader
    trans_test = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[1])
    ])

    D_test = model_util.My_dataset('TMP', 'test.txt', transform=trans_test)
    testloader = DataLoader(D_test, batch_size=36, shuffle=False, num_workers=0, drop_last=False)

    ## Load network
    my_network = net1.AlexNet(num_classes=2, dropout=0.0)
    my_network.load_state_dict(torch.load(args.model, map_location ='cpu'))
    my_network.eval()

    # time3 = time.time()
    # Evaluate
    iter_loader = iter(testloader)
    I, label, full_position =  next(iter_loader)


    ### Cuda device
    # I = I.to('cuda')
    # my_network.to('cuda')
    # logit = my_network(I)
    # logit = logit.cpu()
    #####
    logit = my_network(I)
    #####
    _, predict = torch.max(logit, 1)

    time4 = time.time()
    # pdb.set_trace()
    Pre = []
    Pre_c = []
    for i in range(36):
        if predict[i].numpy() == 1:
            # print(full_position[i])
            Pre.append(full_position[i])
            Pre_c.append(full_position[i].strip('_')[0])

    Count = []
    for i in range(3):
        Count.append(Pre_c.count(str(i+1)))
    # print('Class frequency: {} {} {}'.format(Count[0], Count[1], Count[2]))

    Largest = np.argmax(np.array(Count))
    # print('Image Label: {}'.format(Largest + 1))
    Select_label = [i for i in Pre if i.startswith(str(Largest+1))]
    return Select_label

if __name__ == '__main__':
    agsps = argparse.ArgumentParser()
    agsps.add_argument('file', type=str)
    agsps.add_argument('--size', type=int, default=227)
    agsps.add_argument('--TemplatePath', default='template_bbx')
    agsps.add_argument('--model', default='best.pt')

    args = agsps.parse_args()

    # Select_label = test_image(args.file, args.size)
    Select_label = test_image(args)
    print(Select_label)