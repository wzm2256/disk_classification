from torch import package
import net1
import torch
import argparse


# agsps = argparse.ArgumentParser()
# agsps.add_argument('model', type=str)
# agsps.add_argument('output', type=str)

# args = agsps.parse_args()


def pack_model(output_name, input_name='best.pt'):
    my_network = net1.AlexNet(num_classes=2, dropout=0.1)
    my_network.load_state_dict(torch.load(input_name))
    my_network.eval()


    path = "pack/{}.pt".format(output_name)
    package_name = "Disk_model"
    resource_name = "model.pkl"


    with package.PackageExporter(path) as exp:
        exp.intern("net1.**")
        exp.save_pickle(package_name, resource_name, my_network)

pack_model('new_best_1210', input_name='best.pt')