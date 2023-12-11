
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
import argparse
import model_util
import torch
import pdb
# import net
import torch.optim as optim
import net1
# import network
import torchvision.models as models


agsps = argparse.ArgumentParser()
agsps.add_argument('--size', type=int, default=227)
agsps.add_argument('--crop_size', type=int, default=200)
# agsps.add_argument('--split', type=float, default=0.8)
agsps.add_argument('--train_bs', type=int, default=64)
agsps.add_argument('--lr_D', type=float, default=1e-4)
agsps.add_argument('--smooth', type=float, default=0.0)
agsps.add_argument('--epoch', type=int, default=10)



args = agsps.parse_args()


############# Prepare dataset
trans_train = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    # transforms.RandomCrop(args.crop_size),
    # transforms.RandomHorizontalFlip(), 
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[1])
])

trans_test = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[1])
])


D_train = model_util.My_dataset('Crop', 'train.txt', transform=trans_train)
D_test = model_util.My_dataset('Crop', 'test.txt', transform=trans_test)

trainloader = DataLoader(D_train, batch_size=args.train_bs, shuffle=True, num_workers=0, drop_last=True)
testloader = DataLoader(D_test, batch_size=args.train_bs, shuffle=False, num_workers=0, drop_last=False)
#############Define Network

###
# my_network = network.mobilenet_v3_small(num_classes=2)
##
my_network = net1.AlexNet(num_classes=2, dropout=0.0)
##
# my_network = models.resnet18(num_classes=2)
###

my_network = my_network.cuda()
optimizer = optim.Adam(my_network.parameters(), lr=args.lr_D)
my_loss = model_util.CrossEntropyLabelSmooth(2, epsilon=args.smooth)

Max_test_acc = 0
for e in range(args.epoch):
    # acc = model_util.Eval(my_network, testloader)
    for i, Data  in enumerate(trainloader):
        I, label, _ = Data
        I_cuda, label_cuda = I.cuda(), label.cuda()
        logit = my_network(I_cuda)
        L = my_loss(logit, label_cuda)

        ### tmp acc
        _, predict = torch.max(logit, 1)
        tmp_acc = torch.sum(torch.squeeze(predict).float() == label_cuda).item() / float(label_cuda.size()[0])
        ####
        print(f'Train Epoch {e} Iter {i} \t loss {L:.2f} \t acc {tmp_acc:.2f}.')

        optimizer.zero_grad()
        L.backward()
        optimizer.step()


    my_network.eval()
    acc = model_util.Eval(my_network, testloader)
    if acc >= Max_test_acc:
        torch.save(my_network.state_dict(), 'best.pt')
        Max_test_acc = acc
    
    print('Test------Epoch {}: \t Acc{}'.format(e, acc))
    my_network.train()

print('Final acc: {}'.format(Max_test_acc))