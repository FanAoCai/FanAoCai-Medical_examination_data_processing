import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt
from model import LogisticRegression
from cnn_model import CnnDisease
from tqdm import tqdm
from torchvision.transforms import functional as F
import pandas as pd
from torch.utils.data import WeightedRandomSampler

class Train():
    # def __init__(self):
    def __init__(self,
                 train_model,
                 learning_rate,
                 batch_size,
                 loss_fn,
                 optimizer
                 ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('训练使用的device为:{}'.format(self.device))
        self.train_model = train_model.to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.feature_title = [
            'Normal',
            'Hypertension',
            'Diabetes',
            'Coronary_heart_disease',
            'Asthma',
            'Hyperlipidemia',
            'Hypothyroidism',
            'Gout',
            'High_blood_sugar',
            'Chronic_bronchitis',
            'Cerebral_Infarction',
            'Chronic_gastritis',
            'Fatty_liver',
            'Parkinson_disease',
            'Heart_disease'
        ]

    def nomalize(self, inputdata):
        min = np.nanmin(inputdata)
        max = np.nanmax(inputdata)
        outputdata = (inputdata-min)/(max-min)
        return outputdata
    

    def prepare_feature_selection_data(self, train_split, val_split, order, end):
        origin_data = np.load('data/all_disease/all_disease.npz', allow_pickle=True)
        change_data = np.append(origin_data[self.feature_title[0]],np.zeros((len(origin_data[self.feature_title[0]]),1)), axis=1)
        prepare_data = change_data[0:50000,7:].astype(np.float64)
        frame_title = origin_data['title'][7:]
        # print('length of frame_title:{}'.format(len(frame_title)))
        print('此次训练使用的特征是第{}个，具体为：{}'.format(order, frame_title[order]))
        for idx, feature in tqdm(enumerate(self.feature_title)):
            if idx == 0:
                continue
            elif idx == 1:
                change_data = np.append(origin_data[feature],np.full((len(origin_data[feature]),1),idx),axis=1)
                prepare_data = np.append(prepare_data, change_data[:,7:].astype(np.float64), axis=0)
            else:
                break
        # feature_data = prepare_data[:,0:1].squeeze(1)
        np.random.shuffle(prepare_data)
        feature_data = prepare_data[:,40:len(frame_title)-1]
        # feature_data = self.nomalize(feature_data)
        label_data = prepare_data[:,-1]
        # unique, counts = np.unique(label_data, return_counts=True)
        # print(f'Class distribution: {dict(zip(unique, counts))}')
        label_data = np.eye(2)[label_data.astype('int64')]

        
        train_fraction = int(len(feature_data) * train_split)
        val_fraction = int(len(feature_data) * val_split)
        
        x_train = torch.FloatTensor(feature_data[:train_fraction]).to(self.device)
        y_train = torch.FloatTensor(label_data[:train_fraction]).to(self.device)
        x_val = torch.FloatTensor(feature_data[train_fraction :]).to(self.device)
        y_val = torch.FloatTensor(label_data[train_fraction : ]).to(self.device)
        x_test = torch.FloatTensor(feature_data[(train_fraction + val_fraction) :]).to(self.device)
        y_test = torch.FloatTensor(label_data[(train_fraction + val_fraction) : ]).to(self.device)
        

        train_dataset = data.TensorDataset(x_train, y_train)
        val_dataset = data.TensorDataset(x_val, y_val)

        # print(self.batch_size)
        train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True)
        val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=self.batch_size, drop_last=True)

        return train_loader, val_loader, frame_title[order]
    
    def prepare_all_feature_data(self, train_split, val_split):
        origin_data = np.load('data/all_disease/all_disease.npz', allow_pickle=True)
        change_data = np.append(origin_data[self.feature_title[0]],np.zeros((len(origin_data[self.feature_title[0]]),1)), axis=1)
        prepare_data = change_data[0:50000,7:].astype(np.float64)
        frame_title = origin_data['title'][7:]
        for idx, feature in tqdm(enumerate(self.feature_title)):
            if idx == 0:
                continue
            elif idx == 1:
                change_data = np.append(origin_data[feature],np.full((len(origin_data[feature]),1),idx),axis=1)
                prepare_data = np.append(prepare_data, change_data[:,7:].astype(np.float64), axis=0)
            else:
                break
        np.random.shuffle(prepare_data)
        # feature_data = self.nomalize(feature_data)
        # feature_data = prepare_data[:,:len(frame_title)-1]
        feature_data = prepare_data[:,50:55]
        label_data = prepare_data[:,-1]
        unique, counts = np.unique(label_data, return_counts=True)
        print(f'Class distribution: {dict(zip(unique, counts))}')
        label_data = np.eye(2)[label_data.astype('int64')]

        train_fraction = int(len(feature_data) * train_split)
        val_fraction = len(feature_data) - train_fraction
        
        x_train = torch.FloatTensor(feature_data[:train_fraction]).to(self.device)
        y_train = torch.FloatTensor(label_data[:train_fraction]).to(self.device)
        x_val = torch.FloatTensor(feature_data[train_fraction : ]).to(self.device)
        y_val = torch.FloatTensor(label_data[train_fraction : ]).to(self.device)
        x_test = torch.FloatTensor(feature_data[(train_fraction + val_fraction) :]).to(self.device)
        y_test = torch.FloatTensor(label_data[(train_fraction + val_fraction) : ]).to(self.device)

        train_dataset = data.TensorDataset(x_train, y_train)
        val_dataset = data.TensorDataset(x_val, y_val)

        # print(self.batch_size)
        train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=self.batch_size, drop_last=True)
        val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, drop_last=True)

        return train_loader, val_loader

    def prepare_disease_data(self):

        return
    
    def train_loop(self, dataloader):
        size = len(dataloader.dataset)
        self.train_model.train()
        for step, (x_train, y_train) in enumerate(dataloader):
            output = self.train_model(x_train)         #调用模型预测
            loss = self.loss_fn(output, y_train)    #计算损失值
            # print(loss.item())
            self.optimizer.zero_grad()   #每一次循环之前，将梯度清零
            loss.backward()         #反向传播
            self.optimizer.step()        #梯度下降
            if step % 4000 == 0:
                loss, current = loss.item(), step * len(x_train)
                print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    # def val_loop(self, dataloader):
    #     size = len(dataloader)
    #     val_loss, correct = 0, 0
    #     self.train_model.eval()
    #     print(size)
        
    #     with torch.no_grad():
    #         for x, y in dataloader:
    #             pred = self.train_model(x)
    #             val_loss += self.loss_fn(pred, y).item()
    #             # print(val_loss)
    #             correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
    #             # print(torch.argmax(pred, dim=1))
    #             # print((torch.argmax(pred, dim=1) == y).type(torch.float).sum().item())
    #             # print(y)
    #         val_loss /= size
    #         correct /= (size * self.batch_size)
    #         # print(val_loss)
    #         # print(correct)
    #         print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Val loss: {val_loss:>8f} \n")
        
    #     return val_loss, correct
                
    def val_loop(self, dataloader):
        size = len(dataloader)
        val_loss, correct = 0, 0
        self.train_model.eval()
        
        with torch.no_grad():
            for x, y in dataloader:
                pred = self.train_model(x)
                # print(y)
                # print(pred)
                val_loss += self.loss_fn(pred, y).item()
                pred = (pred >= 0.5).float()
                # print(pred)
                correct += (pred == y).float().mean()
            val_loss /= size
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Val loss: {val_loss:>8f} \n")
        
        return val_loss, correct
        
 

if __name__ == "__main__":

    #3、针对所有特征进行疾病预测
    # train_model = CnnDisease()
    # learning_rate = 0.0001
    # batch_size = 64
    # epochs = 200
    # loss_fn = nn.BCELoss()
    # optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    # train = Train(train_model, learning_rate, batch_size, loss_fn, optimizer)
    # train_loader, val_loader = train.prepare_all_feature_data(0.7, 0.3)
    # size = len(train_loader.dataset)

    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train.train_loop(train_loader)
    #     train.val_loop(val_loader)


    #2、循环针对每个特征进行疾病预测
    # train_model = LogisticRegression()
    # learning_rate = 0.0001
    # batch_size = 128
    # epochs = 100
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    # val_loss_list = []
    # correct_list = []
    # frame_title_list = []
    # for i in range(5, 7):
    #     train = Train(train_model, learning_rate, batch_size, loss_fn, optimizer)
    #     train_loader, val_loader, frame_title = train.prepare_feature_selection_data(0.6, 0.2, i, i+1)
    #     frame_title_list.append(frame_title)
    #     size = len(train_loader.dataset)
    #     for t in tqdm(range(epochs)):
    #         if t==99:
    #             print(f"Epoch {t+1}\n-------------------------------")
    #         train.train_loop(train_loader)
    #         val_loss, correct = train.val_loop(val_loader)
    #         if t == 99:
    #             val_loss_list.append(val_loss)
    #             correct_list.append(correct)
    #             print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Val loss: {val_loss:>8f} \n")
    # data_dict = {
    #     'title': frame_title_list,
    #     'loss': val_loss_list,
    #     'auc': correct_list
    # }
    # save_data = pd.DataFrame(data_dict)
    # save_data.to_csv('feature.csv')


    #1、针对单一特征进行疾病预测
    train_model = LogisticRegression()
    learning_rate = 0.0001
    batch_size = 64
    epochs = 100
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    train = Train(train_model, learning_rate, batch_size, loss_fn, optimizer)
    order = 0
    end = order + 55
    train_loader, val_loader, _ = train.prepare_feature_selection_data(0.7, 0.3, order, end)
    size = len(train_loader.dataset)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train.train_loop(train_loader)
        train.val_loop(val_loader)
        # if (t + 1) % 10 == 0:
        # torch.save(model, 'model_pth/' + str(t+1) + 'model.pth')
    

    


    


