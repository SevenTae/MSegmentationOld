from tqdm import tqdm
import os
'''tqdm的使用'''
source_label_path = r"/data/VOCdevkit/berlin/labelss"
source_label=os.listdir(source_label_path)
for i in tqdm(source_label):
    print(i)