import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import math
from collections import Counter
import re
from matplotlib import pyplot as plt
import seaborn as sns



class DataProcessing():
    def __init__(self,
                csv_file_name):
        super().__init__()

        self.csv_file_name = csv_file_name
        self.all_data = pd.read_csv(self.csv_file_name, low_memory=False)
        self.data_length = len(self.all_data)

        self.feature = ['运动习惯', '饮食习惯1', '饮酒史', '药物过敏史', '现病史', 
                        '吸烟史', '手术史', '既往病史']
        self.exercise_habit = {
            '无',             #0:7379
            '偶尔',           #1:745794
            '经常',           #2:86458
            '每日'            #3:5858
            }
        self.eat_habit = [
            '清淡',             #0:26319
            '一般',             #1:764803
            '辛辣',             #2:25308
            '偏咸',             #3:11842
            '甜食',             #4:2722
            '偏咸；辛辣',        #5:4450
            '辛辣；甜食',        #6:997
            '偏咸；甜食',        #7:22
            '清淡；甜食',        #8193
            '偏咸；辛辣；甜食'    #9:192
        ]
        self.drink = [
            '无',      #0:551137
            '偶尔',    #:1234337
            '经常',    #2:9744
            '每日',    #3:199
            '戒酒'     #4:467
        ]
        self.medicine = {
            '无'
            '青霉素类'
            '磺胺类'
            '头孢类'
            '链霉素类'
            '破伤风'
            '喹诺酮类'
            '甲硝唑'
            '庆大霉素'
            '红霉素'
            '左氧氟沙星'
            '去痛片'
            '沙星类'
            '阿司匹林'
            '四环素'
            '氧氟沙星'
            '阿莫西林'
            '阿奇霉素'
        }
        self.disease_now = [
            '无特殊',       #0:753786:570597:normal        
            '高血压',       #1:63087:44606:hypertension         
            '糖尿病',       #2:22729:16128:diabetes     
            '冠心病',       #3:5252:3628:coronary_heart_diseas       
            '哮喘',         #4:955:664:asthma           
            '高脂血症',     #5:2585:1784:hyperlipidemia     
            '甲减',         #6:1750:1162:hypothyroidism         
            '痛风',         #7:2642:1803:gout         
            '血糖高',       #8:594:444:high_blood_sugar        
            '慢支炎',       #9:697:508:chronic_bronchitis        
            '脑梗',         #10:409:228:cerebral_Infarction        
            '慢性胃炎',     #11:870:722:chronic_gastritis   
            '脂肪肝',       #12:396:322:fatty_liver_data       
            '帕金森',       #13:261:170:parkinson_disease_data    
            '心脏病'        #14:441:258:heart_disease      
        ]
        self.smoke = {

        }
        self.operation ={

        }
        self.disease_before = {

        }

    def judge_nan(self, data):
        try:
            if math.isnan(data) == True:
                result = True
            else:
                result = False
        except:
            result = False
        return result

    def number_nan(self, list_name):
        order = list(map(self.judge_nan, list_name))
        count_result = Counter(order)
        return count_result
    
    def read_data(self):
        print('Start reading data!')
        title = list(self.all_data)
        result_list = {}
        for i in tqdm(title):
            # print(self.all_data[i].nunique())
            number = self.number_nan(self.all_data[i])
            time = dict(number.most_common(len(number)))  #将统计好的counter数据类型转化为list,再转化为dict
            try:
                false_time = time[False]
            except:
                false_time = 0
            print('{}中共有{}/{}个数据, 数据占比为：{}'.format(i, false_time, self.data_length , false_time/self.data_length))
            result_list[i] = false_time / self.data_length   #字典中存储的是每项特征非nan值的数量
        print('Finish preading data!\n')
        return result_list
    
    def judge_delete(self):
        every_time = self.read_data()

        #删除包含较多nan值的特征
        print('Start deleting nan data!')
        print('处理前的特征总量为：{}'.format(len(every_time)))
        delete_list = []
        for idx, time in tqdm(enumerate(every_time)):
            if every_time[time] < 0.80:
                delete_list.append(time)
        after_delete_data = self.all_data.drop(delete_list, axis=1)
        print('处理后的特征总量为：{}'.format(len(every_time)-len(delete_list)))
        print('Finish deleting nan data!\n')

        # #删除现病史与吸烟史中的括号中的内容
        # for idx, every in tqdm(enumerate(after_delete_data['现病史'])):
        #     result = re.sub(r'\(.*\)', "",every)
        #     after_delete_data['现病史'][idx] = result

        #删除数据量缺失的患者
        print('Start deleting patients!')
        array_data = np.array(after_delete_data)
        title = list(after_delete_data)   #将所有列名转化为一个list,留作后续使用
        print('处理前的数据总量为：{}'.format(len(after_delete_data)))
        patient_delete_list = []
        for idx, patient in tqdm(enumerate(array_data)):
            number = self.number_nan(patient)
            time = dict(number.most_common(len(number)))
            if time[False]/len(patient) < 1:
                patient_delete_list.append(idx)
        cancel_data = np.delete(array_data, patient_delete_list, axis=0)
        print('处理后的数据总量为：{}'.format(len(cancel_data)))
        print('Finish deleting patients!\n')
        
        return cancel_data, title  #将少于总数80%的特征删除

    #for test
    def get_train_data(self):
        data_array, frame_title = self.judge_delete()   #正式代码中的内容，测试时不需要
        frame_data = pd.DataFrame(data_array, columns=frame_title)

        # frame_data.to_csv('data/all_disease/alldata_after_delete.csv', encoding="utf_8_sig")

        # 画所有疾病的皮尔逊相关性热图
        # data = frame_data.iloc[:,7:].astype(float).corr()
        # data.to_csv('data/all_disease/all_disease_pearsonheatmap.csv', encoding="utf_8_sig")
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
        # plt.subplots(figsize = (30,30))
        # # sns.heatmap(data, annot = True, vmin=0, vmax = 1, square = True, cmap = "Reds")
        # sns.heatmap(data, annot = False, vmin=0, vmax = 1, square = True, cmap = "Blues")
        # plt.savefig('picture/all_disease/alldisease_pearsonheatmap.png')
        # plt.show()

        #提取每种疾病的患者并保存
        ready_data = []
        feature_sum = []
        for indx, disease in tqdm(enumerate(self.disease_now)):
            delete_pressure_list = []
            for idx, every_data in tqdm(enumerate(frame_data['现病史'])):
                if disease in every_data:
                    continue
                else:
                    delete_pressure_list.append(idx)
            after_data = frame_data.drop(delete_pressure_list, axis=0)
            ready_data.append(np.array(after_data))
            feature_sum.append(len(np.array(after_data)))

        #画所有疾病的柱状图
        # plt.title('疾病柱状图')
        # plt.bar(self.feature_title, feature_sum)
        # plt.show()

        np.savez_compressed('data/all_disease/all_disease.npz', 
                            title = frame_title,
                            Normal = ready_data[0],
                            Hypertension = ready_data[1],
                            Diabetes = ready_data[2],
                            Coronary_heart_disease = ready_data[3],
                            Asthma = ready_data[4],
                            Hyperlipidemia = ready_data[5],
                            Hypothyroidism = ready_data[6],
                            Gout = ready_data[7],
                            High_blood_sugar = ready_data[8],
                            Chronic_bronchitis = ready_data[9],
                            Cerebral_Infarction = ready_data[10],
                            Chronic_gastritis = ready_data[11],
                            Fatty_liver = ready_data[12],
                            Parkinson_disease = ready_data[13],
                            Heart_disease = ready_data[14]
                            )
        return frame_title


        

if __name__ == "__main__":

    dp = DataProcessing('data/all_disease/allrawdata.csv')
    dp.get_train_data()

    # all_data = pd.read_csv('data/all_disease/allrawdata.csv', low_memory=False)
    
    # data = Counter(all_data['甘油三脂'])
    # print(data)

    # data = Counter(all_data['白细胞'])
    # print(data)

    

    