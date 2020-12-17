import imagehash
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
import os
import torch
import time
import matplotlib.pyplot as plt

funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]
image_ids = []
hashes = []
base_path = '/home/user/dataset/gunzi/mmclass_format/all'


# 1、加载所有图片，转化hash

# #for path in tqdm(glob.glob(str(BASE_DIR/'train_images'/'*.jpg' ))):
# for path in tqdm(glob.glob( os.path.join(base_path, '*.jpg') )):
#     image = Image.open(path)
#     image_id = os.path.basename(path)
#     image_ids.append(image_id)
#     hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
# hashes_all = np.array(hashes)
# ### save time, write np to file
# np.save( 'hash_data.npy', hashes_all )

# 或者从本地读取npy，因为上一步时间很长，如果多次调试不要每次都从头开始

for path in tqdm(glob.glob( os.path.join(base_path, '*.jpg') )):
    image_id = os.path.basename(path)
    image_ids.append(image_id)
hashes_all = np.load('hash_data.npy')
hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()


# find duplicated imgs
start = time.time()
sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).cpu().numpy()/256 for i in range(hashes_all.shape[0])])
indices1 = np.where( sims > 0.997)   # clw note: threshold is important：(8834x256), 0.9: found 18706012 duplicates   0.98: found 6690 duplicates
                                                                                    #0.995: found 16 duplicates      0.997:  found 3 duplicates
indices2 = np.where(indices1[0] != indices1[1])
image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([image_ids1,image_ids2])):True for image_ids1, image_ids2 in zip(image_ids1, image_ids2)}
print('found %d duplicates' % len(dups))
print('find duplicated imgs time use: %.3fs' % (time.time() - start))


# Plotting the duplicate Images
# code taken from https://www.kaggle.com/nakajima/duplicate-train-images?scriptVersionId=47295222
duplicate_image_ids = sorted(list(dups))
#######################
### 全部展示
fig, axs = plt.subplots(  len(dups), 2, figsize=(15,15))
for row in range( len(dups)  ):
### 只展示前两张
# fig, axs = plt.subplots(  2, 2, figsize=(15,15))
# for row in range( 2  ):
######################
    for col in range(2):
        img_id = duplicate_image_ids[row][col]
        # img = Image.open(str(BASE_DIR/'train_images'/img_id))
        img = Image.open(os.path.join(base_path,  str(img_id) )  )
        # label =str(train.loc[train['image_id'] == img_id].label.values[0])
        axs[row, col].imshow(img)
        axs[row, col].set_title("image_id : "+ img_id)
        axs[row, col].axis('off')
plt.show()