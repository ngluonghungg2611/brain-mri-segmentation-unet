from A_libraries import *
train_files = []
mask_files = glob('lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))

# print(train_files[:10])
# print(mask_files[:10])