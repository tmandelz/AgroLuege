import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=1,
                     bias=True,
                     padding_mode='replicate')

def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=kernel_size//2,
                     bias=False,
                     padding_mode='replicate')



class model_2DConv(torch.nn.Module):
    def __init__(self, nclasses=36, num_classes_l1=6, num_classes_l2=20, s1_2_s3=None, s2_2_s3=None,
                 planes=128, padding=False, wo_softmax=True, dropout=0.5):
        super(model_2DConv, self).__init__()
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.nclasses = nclasses
        self.hidden_dim = 2 * nclasses
        self.planes = planes
        self.padding = padding
        self.wo_softmax = wo_softmax

        self.s1_2_s3 = s1_2_s3
        self.s2_2_s3 = s2_2_s3

        if self.padding:
            input_dim = self.nclasses*3
        else:
            input_dim = self.nclasses + self.num_classes_l1 + self.num_classes_l2

        #input_dim = self.nclasses
        self.conv1 = conv3x3(input_dim, self.planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(self.planes, self.planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = conv3x3(self.planes, self.planes, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(self.planes)
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = conv3x3(self.planes, self.nclasses, kernel_size=1)


    def forward(self, x):
        x1_, x2_, x3 = x

        # x1_ = F.softmax(x1_, dim=1)
        # x2_ = F.softmax(x2_, dim=1)
        # x3 = F.softmax(x3, dim=1)

        #padding
        if self.padding:
            x1 = Variable(torch.zeros_like(x3), requires_grad=True).cuda()
            x2 = Variable(torch.zeros_like(x3), requires_grad=True).cuda()

            with torch.no_grad():
                for i in range(self.s1_2_s3.shape[0]):
                    x1[:, i,:,:] = x1_[:, int(self.s1_2_s3[i]),:,:]
                    x2[:, i,:,:] = x2_[:, int(self.s2_2_s3[i]),:,:]

            x_concat = torch.cat((x1, x2, x3), dim=1)

        else:
            x_concat = torch.cat((x1_, x2_, x3), dim=1)


        out1 = self.conv1(x_concat)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out = self.conv2(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += out1
        out = self.relu(out)
        out = self.drop3(out)

        out = self.conv4(out)

        out += x3


        if self.wo_softmax:
            return out
        else:
            return F.log_softmax(out, dim=1)


