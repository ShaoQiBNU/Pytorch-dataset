###################### load packages ########################
import torch
import torch.utils.data
import Dataset
import model
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd

###################### 参数设置 ########################
data_path = "iris.csv"
save_path = "iris_pred.csv"

batch_size = 20
epoch = 10

input_size = 4
hidden_size = 4
num_classes = 3


###################### load data ########################
dataset_train = Dataset.IrisDataset(data_path, train=True)
dataset_test = Dataset.IrisDataset(data_path, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)


################# train #################
def train(train_loader, model, optimizer, cost, epoch):

    print("Start training:")

    ########### epoch ##########
    for i in range(epoch):
        train_correct = 0
        total_cnt = 0

        ############## batch #############
        for batch_idx, (data, target) in enumerate(train_loader):

            ############ get data and target #########
            data, target = Variable(data), Variable(target)

            ############ optimizer ############
            optimizer.zero_grad()

            ############ get model output ############
            output = model(data)

            ############ get predict label ############
            _, pred = torch.max(output.data, 1)

            ############ loss ############
            loss = cost(output, target)
            loss.backward()

            ############ optimizer ############
            optimizer.step()

            ############ result ############
            total_cnt += data.data.size()[0]
            train_correct += torch.sum(pred == target.data)

            ############ show train result ############
            if (batch_idx+1) % 10 == 0:
                print("epoch: {}, batch_index: {}, train loss: {:.6f}, train correct: {:.2f}%".format(
                    i, batch_idx+1, loss, 100*train_correct/total_cnt))

    print("Training is over!")


################# test #################
def test(test_loader, model, cost, save_path):

    print("Start testing:")

    pred_res=[]
    label_res=[]
    ############ batch ############
    for batch_idx, (data, target) in enumerate(test_loader):

        ############ get data and target ############
        data, target = Variable(data), Variable(target)

        ############ get model output ############
        output = model(data)


        ############ get predict label ############
        _,pred = torch.max(output.data, 1)

        ############ loss ############
        loss = cost(output, target)

        ############ accuracy ############
        test_correct = torch.sum(pred == target.data)

        ############ 结果保存 ############
        pred_res.extend([t.item() for t in pred])
        label_res.extend([t.item() for t in target.data])

        print("batch_index: {}, test loss: {:.6f}, test correct: {:.2f}%".format(
                batch_idx + 1, loss.item(), 100*test_correct/data.data.size()[0]))

    res = pd.DataFrame()
    res['pred']=pred_res
    res['label'] = label_res
    res.to_csv(save_path, index=False)

    print("Testing is over!")


###################### model ########################
fcn = model.NeuralNet(input_size, hidden_size, num_classes)

################# optimizer and loss #################
optimizer = optim.Adam(fcn.parameters(), lr=0.001)
cost = nn.CrossEntropyLoss()

################# train #################
train(data_loader_train, fcn, optimizer, cost, epoch)

################# test #################
test(data_loader_test, fcn, cost, save_path)