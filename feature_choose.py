import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns

class FeatureChoose():
    def __init__(self,
                 npz_file_name):
        super().__init__()

        self.disease_data = np.load(npz_file_name, allow_pickle=True)
        self.frame_title = self.disease_data['title']

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

    def feature_process(self):
        # #画每个疾病的皮尔逊相关热力图
        for idx, feature in tqdm(enumerate(self.feature_title)):
            data = pd.DataFrame(self.disease_data[feature], columns=self.frame_title).iloc[:,7:].astype(float).corr()
            data.to_csv('data/singal_disease/{}_pearsonheatmap.csv'.format(feature), encoding="utf_8_sig")
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
            plt.subplots(figsize = (30,30))
            # sns.heatmap(data, annot = True, vmin=0, vmax = 1, square = True, cmap = "Reds")
            sns.heatmap(data, annot = False, vmin=0, vmax = 1, square = True, cmap = "Reds")
            plt.savefig('picture/singal_disease/{}_pearsonheatmap.png'.format(feature))
            plt.show()
        # print(data.dtypes)
        # print(len(np.array(data)))
        # data_array = np.array(data)
        # sum = 0
        # first = len(data_array)
        # second = len(data_array[0])
        # for i in range(first):
        #     for j in range(second):
        #         if data_array[i][j] >= 0.5:
        #             sum += 1
        # print((sum-56)/2)
        
        
        # frame_data = pd.read_csv('data/test.csv', low_memory=False)
        # data = frame_data.iloc[:,7:].astype(float).corr()
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
        # plt.subplots(figsize = (12,12))
        # sns.heatmap(data, annot = True, vmin=0, vmax = 1, square = True, cmap = "Reds")
        # plt.show()
        # plt.savefig('picture/Pearson_correlation_coefficient_heatmap.png')
            
    def feature_choose(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('训练使用的device为:{}'.format(device))




if __name__ == "__main__":

    fc = FeatureChoose('data/all_disease.npz')
    np.set_printoptions(threshold=np.inf)
    # fc.feature_process()
