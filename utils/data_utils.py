import re

import config as cfg
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
from torchvision import transforms
from sklearn.utils import shuffle
import os
import random
import glob
from PIL import Image
from utils.Utils import set_torch_seed

label_names_en = ['Education',
                  'Product',
                  'Promotion',
                  'Introduction',
                  'Festival',
                  'Real estate',
                  'New house',
                  'Exhibition',
                  'Others']

major_label_names_en = ['Education',
                        'Product',
                        'Real estate',
                        'Exhibition',
                        'Others']

label_names = ['教學',
               '產品',
               '促銷',
               '介紹',
               '節慶',
               '房地產',
               '新成屋',
               '展演',
               '其它']

major_label_names = ['教學',
                     '產品',
                     '房地產',
                     '展演',
                     '其它']

major_indexes = [0, 1, 5, 7, 8]

processor_names = [
    'BeitImageProcessor',
    'ConvNextImageProcessor',
    'SegformerImageProcessor',
    'ViTImageProcessor',
    'ImageGPTImageProcessor',
    'EfficientNetImageProcessor'
]

timm_processor_names = [
    'Compose'
]

folder_name_map = {'train': 'training', 'val': 'valid', 'test': 'test'}

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def auto_split_data(image_list, label_list, setname):
    train_n = int(0.7 * len(image_list))
    imgs_train, imgs_test = image_list[:train_n], image_list[train_n:]
    labels_train, labels_test = np.split(label_list, [train_n])
    if setname == 'train':
        train_n = int(0.9 * len(imgs_train))
        image_list = imgs_train[:train_n]
        label_list, _ = np.split(labels_train, [train_n])
    elif setname == 'val':
        train_n = int(0.9 * len(imgs_train))
        image_list = imgs_train[train_n:]
        _, label_list = np.split(labels_train, [train_n])
    elif setname == 'test':
        image_list = imgs_test
        label_list = labels_test
    return image_list, label_list


def get_transformed_image(images, transform):
    trans_name = type(transform).__name__
    print(f'Transforming data... name: {trans_name}')
    if trans_name == 'ImageGPTImageProcessor':
        data = torch.stack([transform(img, return_tensors="pt").input_ids[0] for img in tqdm(images)])
    elif trans_name in processor_names:
        data = torch.stack([transform(img, return_tensors="pt").pixel_values[0] for img in tqdm(images)])
    elif trans_name in timm_processor_names:
        data = torch.stack([transform(img) for img in tqdm(images)])
    else:
        images = images.astype(float)
        # data = [torch.from_numpy(np.array(img)) for img in images]
        data = [torch.permute(transform(torch.from_numpy(np.array(img))), (1, 2, 0)) for img in tqdm(images)]
    return data, images


class ImageClsDataset(Dataset):
    def __init__(
            self, dataset_path, setname='train', transform=transformation,
            img_ext='jpg', data_size=None, split_data=True
    ):
        self.setname = setname
        assert setname in ['train', 'val', 'test']

        if split_data:
            image_folder = os.path.join(dataset_path, folder_name_map[setname], 'images')
            label_folder = os.path.join(dataset_path, folder_name_map[setname], 'labels')
        else:
            image_folder = os.path.join(dataset_path, 'images')
            label_folder = os.path.join(dataset_path, 'labels')
        self.image_list = list(sorted(glob.glob(os.path.join(image_folder, f'*.{img_ext}')), key=len))
        self.label_list = list(sorted(glob.glob(os.path.join(label_folder, '*.json')), key=len))
        if not split_data:
            self.image_list, self.label_list = auto_split_data(self.image_list, self.label_list, setname)

        self.images, self.labels = load_data_by_list(self.image_list, self.label_list, data_size=data_size)
        self.images, self.labels = shuffle(self.images, self.labels, random_state=cfg.SEED)
        self.transform = transform
        self.data, self.images = get_transformed_image(self.images, self.transform)
        self.classes = label_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.data[idx]
        # if self.transform and \
        #         type(self.transform).__name__ not in processor_names and \
        #         type(self.transform).__name__ not in timm_processor_names:
        #     image = self.transform(image)
        #     image = torch.permute(image, (1, 2, 0))

        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label


def load_label_definitions(filename):
    with open(filename, encoding='utf-8') as f:
        text_list = f.read().strip().replace('\t', '').split('\n')
    res = {text_list[i]: i for i in range(len(text_list))}
    return res


flag_dict = load_label_definitions(cfg.flag_path)
class_num = len(flag_dict)


def get_yolo_transform(img_size, do_normalize=True):
    res = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    if do_normalize:
        return transforms.Compose([
            res,
            transforms.Normalize([0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    return res


def load_label(file_path):
    res = np.zeros(class_num, dtype=np.uint8)
    with open(file_path, encoding='utf-8') as f:
        js = json.load(f)['flags']
    true_dict = {k: v for k, v in js.items() if v}
    for true in true_dict.keys():
        if true in label_names:
            res[flag_dict[true]] = 1
    return res


def load_labels(file_fold, load_major_label=False):
    res = []
    for file in tqdm(glob.glob(file_fold + '/*.json')):
        res.append(load_label(file))
    if load_major_label:
        major_index = [i for i, v in enumerate(major_label_names) if v in label_names]
        res = list(map(lambda x: [v for i, v in enumerate(x) if i in major_index], res))
    return res


def load_data_by_folder(
        img_folder, label_folder, need_path=False,
        ext='jpg', data_size=None, has_label=True
):
    if data_size:
        img_list = list(filter(lambda x: ext in x, sorted(os.listdir(img_folder), key=len)))[:data_size]
    else:
        img_list = list(filter(lambda x: ext in x, sorted(os.listdir(img_folder), key=len)))
    imgs = []
    labels = []
    img_paths = []
    for file in tqdm(img_list):
        img = Image.open(os.path.join(img_folder, file))
        # img = cv_imread(os.path.join(img_folder, file))
        imgs.append(img)
        if need_path:
            fp = os.path.join(img_folder, f'{file.split(".")[0]}.{ext}')
            img_paths.append(fp)
        file_name = file.split('.')[0]
        if has_label:
            label = load_label(os.path.join(label_folder, f'{file_name}.json'))
            labels.append(label)
    # imgs = np.array(imgs)
    labels = np.array(labels)
    if need_path:
        return imgs, labels, img_paths
    return imgs, labels


# 讀取中文路徑圖片
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def load_data_by_list(
        img_list, label_list, need_path=False, data_size=None
):
    if data_size:
        img_list = img_list[:data_size]
    images = []
    labels = []
    img_paths = []
    for image, label in tqdm(zip(img_list, label_list), total=len(img_list)):
        img = Image.open(image)
        # img = cv_imread(image)
        images.append(img)
        label = load_label(label)
        labels.append(label)
        if need_path:
            img_paths.append(image)
    labels = np.array(labels)
    if need_path:
        return images, labels, img_paths
    return images, labels


def is_split_data(dataset_path):
    return set(folder_name_map.values()).issubset(set(os.listdir(dataset_path)))


def get_dataset(dataset_path=cfg.dataset_path, img_ext='jpg',
                transform=transformation, data_size=None, setname=None):
    set_torch_seed()
    split_data = is_split_data(dataset_path)
    if split_data:
        print('Loading split dataset...')
    else:
        print('Loaded unsplit dataset, Now auto split data...')

    def create_dataset(name):
        return ImageClsDataset(dataset_path, name, transform=transform,
                               img_ext=img_ext, data_size=data_size, split_data=split_data)
    if setname:
        return create_dataset(setname)
    train_dataset = create_dataset('train')
    valid_dataset = create_dataset('val')
    test_dataset = create_dataset('test')
    return train_dataset, valid_dataset, test_dataset


def get_dataloader(dataset_path=cfg.dataset_path, img_ext='jpg',
                   transform=transformation, data_size=None, setname=None):
    if transform is None:
        transform = transformation
    set_torch_seed()
    if setname:
        dataset = get_dataset(dataset_path, img_ext,
                              transform, data_size=data_size, setname=setname)
        return DataLoader(dataset, shuffle=bool(setname == 'train'))

    train_dataset, valid_dataset, test_dataset = get_dataset(dataset_path, img_ext, transform,
                                                             data_size=data_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_batch)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch)
    return train_loader, valid_loader, test_loader
