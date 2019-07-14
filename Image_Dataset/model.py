################# load packages #################
import torch.nn as nn
import torch.nn.functional as F

############### main ###########
class GQNet(nn.Module):

    def __init__(self, nums_class=6):

        super(GQNet, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(95, 120, 7, 2),
                                nn.BatchNorm2d(120),
                                nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(120, 240, 5, 2),
                                   nn.BatchNorm2d(240),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(240, 480, 3, 2),
                                   nn.BatchNorm2d(480),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(480, 560, 3, 2),
                                   nn.BatchNorm2d(560),
                                   nn.ReLU(inplace=True))

        self.avgpool=nn.AvgPool2d(10, 10)

        self.fc=nn.Linear(560, nums_class)


    def forward(self, input):

        output=self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output=self.avgpool(output)
        output = output.view(-1, 1 * 1 * 560)
        output=self.fc(output)
        output= F.log_softmax(output, dim=1)
        return output