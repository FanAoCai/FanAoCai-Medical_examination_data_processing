from txt_csv import TxtToCsv
from data_processing import DataProcessing
from feature_choose import FeatureChoose
from tqdm import tqdm, trange

if __name__ == "__main__":
    #将txt文件转化为csv文件进行后续处理
    ttc = TxtToCsv('data/allrawdata.txt')
    csv_file_name = ttc.convert_file_type()

    #对csv文件进行处理，主要操作位删除特征数量少的特征，删除特征信息不全的患者，将剩余的患者按照疾病种类保存到npz文件中用于后续特征筛选
    dp = DataProcessing('data/allrawdata.csv')
    frame_title = dp.get_train_data()

    #对特征进行筛选，减少特征数量，将筛选好的特征保存用于后续训练
    fc = FeatureChoose(frame_title, 'data/all_disease.npz')

    #提取数据进行训练
    


