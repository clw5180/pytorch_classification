import csv
import os

img_root_path = '/home/user/dataset/kaggle2020_leaf_v1/all'
img_sub_folders = os.listdir(img_root_path)

csvFile = open(os.path.join(img_root_path, 'train.csv'), 'w', newline='')

try:
    writer = csv.writer(csvFile)
    writer.writerow(('image_id', 'label'))
    for img_sub_folder in img_sub_folders:
        img_names = os.listdir(os.path.join(img_root_path, img_sub_folder))
        for img_name in img_names:
            writer.writerow((img_name, img_sub_folder))
finally:
    csvFile.close()
