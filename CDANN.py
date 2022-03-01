from sympy import numbered_symbols
from EEGNetMCD import Feature, Predictor
from torch.autograd import Function
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from prepareData import prepareOneSubj, prepareForML
import numpy as np
from scipy.io import savemat
import csv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class CondDomainClassifier(nn.Module):
    def __init__(self,num_domain=10):
        super(CondDomainClassifier, self).__init__()
        self.cd_classifier = nn.Sequential(
            nn.Linear(42,20),
            nn.ReLU(inplace=True),
            nn.Linear(20,20),
            nn.ReLU(inplace=True),
            nn.Linear(20,num_domain)
        )

    def forward(self, x):
        cd_predict = self.cd_classifier(x)
        return cd_predict

class TaskClassifier(nn.Module):
    def __init__(self,num_class=2):
        super(TaskClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(42,20),
            nn.ReLU(inplace=True),
            nn.Linear(20,num_class)
        )

    def forward(self, x):
        predict = self.classifier(x)
        return predict

class CDANN(nn.Module):
    def __init__(self, channel_num=30, num_class=2, num_domain=6, alpha=1):
        super(CDANN, self).__init__()
        self.alpha = alpha
        self.features = Feature(channel_num=channel_num)
        self.task_classifier = TaskClassifier(num_class=num_class)
        self.prior_domain_classifier = CondDomainClassifier(num_domain=num_domain)
        self.cond_domain_classifier = [CondDomainClassifier(num_domain=num_domain) for i in range(num_class)]
        self.GRL = GRL()

    def forward(self, x_y):
        x, y, cd_predicts = x_y[0], x_y[1], x_y[2]
        fea = self.features(x)
        task_predict = self.task_classifier(fea)
        fea = GRL.apply(fea, self.alpha)
        pd_predict = self.prior_domain_classifier(fea)
        for i in range(fea.shape[0]):
            cd_predicts[i] = self.cond_domain_classifier[y[i]](fea[i])
        return task_predict, pd_predict, cd_predicts

def subj_independ(subj_file):

    X_train,y_train,domain_labels = prepareForML(subj_file)
    X_test, y_test = prepareOneSubj(subj_file)
    train_x, test_x = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    train_y, test_y = torch.LongTensor(y_train), torch.LongTensor(y_test)
    domain_labels = torch.LongTensor(domain_labels)
    labels = torch.concat((train_y.unsqueeze(0),domain_labels.unsqueeze(0)))
    test_labels = torch.concat((test_y.unsqueeze(0),domain_labels[0:test_x.shape[0]].unsqueeze(0)))
    train_data = Data.TensorDataset(train_x, labels.T)
    test_data = Data.TensorDataset(test_x, test_labels.T)
    # Hyper Parameters
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    EPOCH = 100
    BATCH_SIZE = 64
    LR = 0.005

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
    model = CDANN(channel_num=5, num_class=2, num_domain=6).to(device)
    for modelet in model.cond_domain_classifier:
        modelet.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(device)

    best_acc = 0

    for epoch in range(EPOCH):
        model.train()
        for step, (x, yy) in enumerate(train_loader):
            b_x = x.to(device)
            b_y = yy[:,0].to(device)
            b_dl = yy[:,1].to(device)
            cd_predicts = torch.rand((b_x.shape[0],6)).to(device)
            task_predict, pd_predict, cd_predicts = model((b_x,b_y,cd_predicts))
            class_loss = loss_func(task_predict, b_y)
            pd_loss = loss_func(pd_predict, b_dl)
            cd_loss = loss_func(cd_predicts, b_dl)
            loss = class_loss + pd_loss + cd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch % 10 == 0:
        print('epoch:',epoch,', loss:',loss.data.cpu().numpy())

        model.eval()
        all, right = 0, 0
        for step, (x, yy) in enumerate(train_loader):
            b_x = x.to(device)
            b_y = yy[:,0].to(device)
            b_dl = yy[:,1].to(device)
            cd_predicts = torch.rand((b_x.shape[0],6)).to(device)
            task_predict, pd_predict, cd_predicts = model((b_x,b_y,cd_predicts))
            pred_train_y = torch.argmax(task_predict,1).data.cpu().numpy()
            all += pred_train_y.shape[0]
            right += np.sum(pred_train_y==b_y.data.cpu().numpy())
        train_acc = right/all

        all, right = 0, 0
        for step, (x, yy) in enumerate(test_loader):
            # print(step)
            b_x = x.to(device)
            b_y = yy[:,0].to(device)
            b_dl = yy[:,1].to(device)
            cd_predicts = torch.rand((b_x.shape[0],6)).to(device)
            task_predict, pd_predict, cd_predicts = model((b_x,b_y,cd_predicts))
            pred_train_y = torch.argmax(task_predict,1).data.cpu().numpy()
            all += pred_train_y.shape[0]
            right += np.sum(pred_train_y==b_y.data.cpu().numpy())
        test_acc = right/all
        
        if best_acc < test_acc:
            best_acc = test_acc

        print('train_acc:',round(train_acc,4),', test_acc:',round(test_acc,4),', best_acc:',round(best_acc,4))

    # save feature

    # save preds
    # savemat('dann_preds/'+subj_file,
    # {
    #     'preds': pred_test_y,
    #     'ground':test_y
    # })

    # savemat('dann_feature/'+subj_file,
    # {
    #     'src_fea':train_fea.detach().cpu().numpy(), 
    #     'tgt_fea':test_fea.detach().cpu().numpy(), 
    #     'src_lab':train_y,
    #     'tgt_lab':test_y
    # })

    return subj_file, test_acc, best_acc


if __name__ == "__main__":
    setup_seed(2)
    independ = []

    f = open('DROZY/CDANN_result.csv',
             'a', encoding='utf=8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['File', 'Train_acc', 'Test_acc'])

    for subj_file in os.listdir("DROZY/drozy/"):
        print("############ %s ##############" %(subj_file))
        subj_file, test_acc, best_acc = subj_independ(subj_file)
        csv_writer.writerow([subj_file, round(test_acc,4), round(best_acc,4)])
        independ.append((subj_file, test_acc, best_acc))

    print("############ ALL FINISHED ##############")
    for (subj_file, test_acc, best_acc) in independ:
        print("%s, %.4f, %.4f" %(subj_file, test_acc, best_acc))