import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import random
import PIL.Image as Image
import pdb
import torch
import torch.nn as nn
from torchvision import transforms

class My_dataset(Dataset):
    def __init__(self, root, file_llist, transform=None, size=1024):
        super().__init__()
        
        self.All = []
        f = open(os.path.join(root, file_llist))
        for line in f.readlines():
            self.All.append(line.strip())
        f.close()

        self.root = root
        self.transform = transform
        self.size = size

        self.to_image = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        name = self.All[index]
        label = int(name.strip('.png').split('_')[-1])
        image_type = name.strip('.png').split('_')[1]
        position = name.strip('.png').split('_')[2]
        full_type = image_type + '_' + position

        with Image.open(os.path.join(self.root, name)) as I:
            # I = I.convert('RGB')
            # # I = I.convert('L')
            # # pdb.set_trace()
            
            # # I = self.transform(I)
            # I = self.to_tensor(I)
            
            # # pdb.set_trace()
            # h = I.shape[1]
            # w = I.shape[2]
            # w_pad = (1024 - w) // 2
            # h_pad = (1024 - h) // 2
            # assert w_pad > 0, 'Width too large'
            # assert h_pad > 0, 'Height too large'
            # Large_tensor = torch.nn.functional.pad(I, (w_pad, w_pad, h_pad, h_pad), mode='constant')
            # Large_image = self.to_image(Large_tensor)

            # if self.transform is not None:
            #     I = self.transform(Large_image)
            I = self.transform(I)
            # pdb.set_trace()
            ###############
            # Type_tensor = torch.nn.functional.one_hot(torch.tensor(int(image_type)-1), num_classes=3).unsqueeze(-1).unsqueeze(-1) * 0.5
            # concat_I = torch.cat([Type_tensor.repeat(1, I.shape[1], I.shape[2]), I], 0)
            # return concat_I, label, full_type
            ###############
            # pdb.set_trace()
            return I, label, full_type

    def __len__(self):
        return len(self.All)






class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        # self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.weight = weight

    def smooth(self, targets):
        targets = torch.zeros((targets.shape[0], self.num_classes)).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        smoothed_targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return smoothed_targets

    def forward(self, inputs, targets, weight=None, norm_type=2):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        smoothed_targets = self.smooth(targets)
        loss = (- smoothed_targets * log_probs).sum(dim=1)
        return loss.mean()


def Eval(model, loader):
    start_test = True
    # with torch.no_grad():
    # iter_test = iter(loader)
    # model_c = model.cpu()
    # pdb.set_trace()
    # for i in range(len(loader)):
    for i, Data  in enumerate(loader):
        I, label, _ = Data
        # data = next(iter_test)
        # inputs = data[0]
        # labels = data[1]
        # inputs = inputs.cuda()
        inputs, labels = I.cuda(), label.cuda()
        # pdb.set_trace()
        outputs = model(inputs)
        
        # outputs = model_c(inputs)
        # pdb.set_trace()
        if start_test:
            all_output = torch.max(outputs.float().cpu(), 1)[1]
            # pdb.set_trace()
            all_label = labels.float().cpu()
            start_test = False
        else:
            all_output = torch.cat((all_output, torch.max(outputs.float().cpu(), 1)[1]), 0)
            all_label = torch.cat((all_label, labels.float().cpu()), 0)

    # _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(all_output).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

