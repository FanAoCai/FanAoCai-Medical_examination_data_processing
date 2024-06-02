import cv2
import numpy as np
import pandas as pd
import os
import chardet

class TxtToCsv():
    def __init__(self,
                txt_file_name):
        super().__init__()
        self.txt_file_name = txt_file_name

    def convert_file_type(self):
        # with open(file_name, 'r', encoding='ANSI') as f: # Encoding way can be seen in the bottle of the file window
        #     for idx,data in enumerate(f.readlines()):
        #         print(len(data.split()))
        
        print('Start converting file type!')
        citys=pd.read_table(self.txt_file_name, encoding='ANSI', header=None, low_memory=False)[:200]  #pd.read_table : read txt file
        csv_file_name = self.txt_file_name.split('.')[0] + '_less' + '.csv'
        print('The name of new csv file is: ' + csv_file_name + '.')
        citys.to_csv(csv_file_name, encoding="utf_8_sig")
        print('Finish converting file type!')
        return csv_file_name

if __name__ == "__main__":
    file_name = os.path.basename('data/allrawdata_less.csv')
    print(file_name)




                                                                                                                                                               