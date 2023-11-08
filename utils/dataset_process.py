import glob
import json
import os
import platform
import re
import shutil
import cv2

from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
from utils import data_utils
from PIL import Image


SEPARATOR = '\\' if platform.system() == 'Windows' else '/'
TARGET_NAME = data_utils.label_names


def get_label_from_json(json_file: str, target_name: list[str]) -> list[str]:
    with open(json_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)['flags']
    true_label = [k for k, v in labels.items() if k in target_name and v]
    return true_label if true_label else None


# 統計分類標籤數量
def statis_dataset_label(label_dir, target_name):
    label_case = [['產品', '促銷', '介紹', '節慶'],
                  ['產品', '促銷'], ['產品', '介紹'],
                  ['產品', '節慶'], ['產品', '促銷', '介紹'],
                  ['產品', '介紹', '節慶'], ['產品', '促銷', '節慶'],
                  ['房地產', '新成屋']]
    label_case.extend([[tar] for tar in target_name])

    label_dict = {
        'case': label_case,
        'count': [0 for _ in range(len(label_case))],
        'path': [[] for _ in range(len(label_case))]
    }
    for label in tqdm(glob.glob(f'{label_dir}/*.json')):
        true_label = get_label_from_json(label, target_name)
        if true_label in label_dict['case']:
            idx = label_dict['case'].index(true_label)
            label_dict['count'][idx] += 1
            label_dict['path'][idx].append(label)
    return label_dict


def move_dataset_by_dict(
        file_dict, img_dir, target_dir,
        img_ext='jpg', label_ext='json',
        copy=True, subdir=True
):
    for i, labels in enumerate(tqdm(file_dict['path'])):
        for j, label in enumerate(labels):
            file_name = label.split(SEPARATOR)[-1].split('.')[0]
            if copy:
                file_process = shutil.copy
            else:
                file_process = shutil.move
            img = f'{file_name}.{img_ext}'
            sub_dir = re.sub('\[|\]|\'|\s', '', str(file_dict['case'][i]))
            target_path = os.path.join(target_dir, sub_dir) if subdir else target_dir
            os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)
            os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
            file_process(label, os.path.join(target_path, 'labels', f'{file_name}.{label_ext}'))
            file_process(os.path.join(img_dir, img), os.path.join(target_path, 'images', img))


def move_dataset_by_labels(
        img_dir: str, label_dir: str, label_names: list[str],
        target_dir: str, file_num: int = 100, img_ext: str = 'jpg',
        label_ext: str = 'json', copy: bool = True, subdir: bool = True
):
    label_counter = [0 for _ in range(len(label_names))]
    labels = shuffle(os.listdir(label_dir), random_state=123)
    pbar = tqdm(total=len(labels))

    if copy:
        file_process = shutil.copy
    else:
        file_process = shutil.move

    while labels and sum(label_counter) < file_num * len(label_names):
        label = labels.pop()
        file_name = label.split('.')[0]
        label_path = os.path.join(label_dir, f'{file_name}.{label_ext}')
        class_name = get_label_from_json(label_path, label_names)
        if class_name is None:
            continue
        class_name = class_name[0]
        class_idx = label_names.index(class_name)
        sub_dir = label_names[class_idx]
        target_path = os.path.join(target_dir, sub_dir) if subdir else target_dir
        os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)
        if label_counter[class_idx] < file_num:
            label_counter[class_idx] += 1
            img_path = os.path.join(img_dir, f'{file_name}.{img_ext}')
            file_process(label_path, os.path.join(target_path, 'labels', f'{file_name}.{label_ext}'))
            file_process(img_path, os.path.join(target_path, 'images', f'{file_name}.{img_ext}'))
        pbar.update(1)
    pbar.close()


def search_file_by_list(file_list, tar_paths, out_fold, copy=True):
    print(file_list)
    if copy:
        process = shutil.copy
    else:
        process = shutil.move
    for file in tqdm(glob.glob(tar_paths)):
        filename = file.split(SEPARATOR)[-1].split('.')[0]
        if filename in file_list:
            process(file, os.path.join(out_fold, file.split(SEPARATOR)[-1]))


def correct_label_imgpath(label_path: str, img_ext: str = 'png'):
    with open(label_path, 'r+', encoding='utf-8') as f:
        js = json.load(f)
        img_paths = js['imagePath'].split(SEPARATOR)
        flags = js['flags']
        if '中古屋' in flags.keys():
            del flags['中古屋']
        if img_paths[-2] == 'image':
            img_paths[-2] = 'images'
        img_file = img_paths[-1]
        if img_file.split('.')[-1] == 'jpg':
            img_paths[-1] = f"{img_file.split('.')[0]}.{img_ext}"
        img_path = SEPARATOR.join(img_paths)
        js['imagePath'] = img_path
        js['flags'] = flags
        f.seek(0)
        f.truncate(0)  # erase file content
        json.dump(js, f, indent=2, ensure_ascii=False)


def correct_label_by_dirs(dirs: list[str]):
    for label_dir in tqdm(dirs):
        for label in glob.glob(os.path.join(label_dir, '*.json')):
            correct_label_imgpath(label)


def create_label_file(img_path, label_path, tar_label, json_format='../Data/json_format.json'):
    with open(json_format, encoding='utf-8') as f:
        js = json.load(f)
    js['imagePath'] = img_path
    js['flags'] = {k: True if k in tar_label else False for k, v in js['flags'].items()}
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=2)


def create_label_files(img_paths, label_paths, tar_label, json_format='../Data/json_format.json'):
    if not img_paths:
        return
    for img_path, label_path in tqdm(zip(img_paths, label_paths), total=len(img_paths)):
        create_label_file(img_path, label_path, tar_label, json_format=json_format)


def remove_empty_label(img_dir, label_dir, label_ext='json'):
    img_set = set(map(lambda x: x.split('.')[0], os.listdir(img_dir)))
    label_set = set(map(lambda x: x.split('.')[0], os.listdir(label_dir)))
    empty_label_set = label_set.difference(img_set)
    cnt = 0
    for file_name in tqdm(empty_label_set):
        if file_name != 'classes':
            os.remove(os.path.join(label_dir, f'{file_name}.{label_ext}'))
            cnt += 1
    print(f'deleted total {cnt} files')


def remove_empty_label_in_folders(folders):
    for folder in folders:
        image_folder = os.path.join(folder, 'images')
        label_folder = os.path.join(folder, 'labels')
        remove_empty_label(image_folder, label_folder)


# 刪除標註檔圖片編碼，回傳計數值
def remove_label_data(label, copy=False):
    open_mode = 'r+'
    if copy:
        open_mode = 'r'
    with open(label, open_mode, encoding='utf-8') as f:
        js = json.load(f)
        if js['imageData']:
            js['imageData'] = None
            if not copy:
                f.seek(0)
                f.truncate(0)  # erase file content
                json.dump(js, f, indent=2, ensure_ascii=False)
            else:
                sep = '/' if '/' in label else '\\'
                file_fold = sep.join(label.split(sep)[:-1])
                file_name = label.split(sep)[-1]
                with open(os.path.join(file_fold, f'copy_{file_name}'), 'w', encoding='utf8') as fw:
                    json.dump(js, fw, indent=2, ensure_ascii=False)
            return 1
        else:
            return 0


def remove_labels_data(label_fold, copy=False):
    cnt = 0
    for label in tqdm(glob.glob(label_fold + '/*.json')):
        cnt += remove_label_data(label, copy)
    print(f'deleted total {cnt} datas.')


def get_dataset_intersection(data1, data2):
    return set(data1).intersection(set(data2))


def get_dataset_difference(data1, data2):
    return set(data1).difference(set(data2))


def get_filenames_by_folder(folder):
    if os.path.exists(folder):
        return list(map(lambda x: x.split('.')[0], os.listdir(folder)))
    else:
        return None


def get_filename_by_path(path):
    return path.split('\\')[-1].split('.')[0]


def get_filenames_by_paths(path_list):
    return list(map(get_filename_by_path, path_list))


def one_hot_to_label_name(encode, target_name):
    if isinstance(encode[0], list):
        return list(one_hot_to_label_name(c, target_name) for c in encode)
    else:
        return list(target for i, target in enumerate(target_name) if encode[i] == 1)


def list_to_str(ll):
    if isinstance(ll, list):
        return ','.join(ll)
    else:
        return ll


# 修改標註檔類別
def edit_label(json_file, tar_label, target_name):
    sor_label = get_label_from_json(json_file, target_name)
    if sor_label != tar_label:
        with open(json_file, 'r+', encoding='utf-8') as f:
            js = json.load(f)
            js['flags'] = {k: True if k in tar_label else False for k, v in js['flags'].items()}
            f.seek(0)
            f.truncate(0)  # erase file content
            json.dump(js, f, indent=2, ensure_ascii=False)
        print(f'edited:{json_file}')
        return 1
    else:
        return 0


def edit_labels(json_dir, tar_label, target_name):
    cnt = 0
    for json_file in tqdm(glob.glob(json_dir + '/*.json')):
        cnt += edit_label(json_file, tar_label, target_name)
    print(f'edited total {cnt} files.')


def classify_dataset_by_case(data_path):
    data_path = r'C:\Users\ray93978\Data\Islab\Drive\Dataset\FB-Ad\classification\dataset\train-review-5'
    img_dir = os.path.join(data_path, 'images')
    label_dir = os.path.join(data_path, 'labels')
    label_names = data_utils.major_label_names
    label_files = statis_dataset_label(label_dir, label_names)
    print(*zip(label_files['case'], label_files['count']), sep='\n')
    target_dir = fr'C:\Users\ray93978\Data\local\label_survey\{data_path.split(SEPARATOR)[-1]}'
    move_dataset_by_dict(label_files, img_dir, target_dir, img_ext='jpg', copy=True)


def get_label_map(file_paths, label_idx=-3):
    label_maps = {}
    for file in file_paths:
        label_name = file.split(SEPARATOR)[label_idx]
        file_name = file.split(SEPARATOR)[-1].split('.')[0]
        if label_name not in label_maps.keys() or not label_maps[label_name]:
            label_maps[label_name] = [file_name]
        else:
            label_maps[label_name].append(file_name)
    return label_maps


def sort_dataset_by_images(dataset_fold, image_ext='jpg', label_ext='json'):
    images = glob.glob(dataset_fold + rf'\*\*\*.{image_ext}')
    labels = glob.glob(dataset_fold + rf'\*\*\*.{label_ext}')
    image_maps = dict(map(lambda x: (x.split(SEPARATOR)[-1].split('.')[0], x.split(SEPARATOR)[-3]), images))
    cnt = 0
    log = []
    for label in tqdm(labels):
        file_name = label.split(SEPARATOR)[-1].split('.')[0]
        file_label = label.split(SEPARATOR)[-3]
        if file_name in image_maps.keys() and file_label != image_maps[file_name]:
            cnt += 1
            new_label_path = label.split(SEPARATOR)
            new_label_path[-3] = image_maps[file_name]
            new_label_path = SEPARATOR.join(new_label_path)
            shutil.move(label, new_label_path)
            log.append(f'moved : {label} -> {new_label_path}')
    print(cnt)
    print(*log, sep='\n')


def move_file_to_target_folder(files, targets, label_folder, tar_folder, label_ext='json'):
    if targets:
        for file in tqdm(get_dataset_intersection(files, targets)):
            shutil.move(os.path.join(label_folder, f'{file}.{label_ext}'),
                        os.path.join(tar_folder, f'{file}.{label_ext}'))


def move_except_file(file_folder):
    bad_folder = os.path.join(file_folder, 'bad')
    same_folder = os.path.join(file_folder, 'same')
    confu_folder = os.path.join(file_folder, 'confusion')
    bad_files = get_filenames_by_folder(os.path.join(bad_folder, 'images'))
    same_files = get_filenames_by_folder(os.path.join(same_folder, 'images'))
    confu_files = get_filenames_by_folder(os.path.join(confu_folder, 'images'))
    label_folder = os.path.join(file_folder, 'label')
    for fold in tqdm(os.listdir(label_folder)):
        if fold != 'bad' and fold != 'same' and fold != 'confusion':
            label_file = get_filenames_by_folder(os.path.join(label_folder, fold, 'labels'))
            sub_label_folder = os.path.join(label_folder, fold, 'labels')
            move_file_to_target_folder(label_file, bad_files, sub_label_folder, os.path.join(bad_folder, 'labels'))
            move_file_to_target_folder(label_file, same_files, sub_label_folder, os.path.join(same_folder, 'labels'))
            move_file_to_target_folder(label_file, confu_files, sub_label_folder, os.path.join(confu_folder, 'labels'))


def sort_survey_data():
    data_folder = r'C:\Users\ray93978\Data\local\label_survey\train-review-6\confusion'
    label_folder = os.path.join(data_folder, 'label')
    move_except_file(data_folder)
    sort_dataset_by_images(data_folder)
    remove_empty_label_in_folders(glob.glob(label_folder + '/*'))
    # remove_empty_label()
    create_missing_label(label_folder)


def main2():
    data1_path = r'C:\Users\ray93978\Data\Islab\Ray93978\AD\AdsImgUtils\data\objects\images'
    data2_path = r'C:\Users\ray93978\Data\Islab\Drive\Local\label_agreement\object detection\labels\annotator_1'
    class_path = r'C:\Users\ray93978\Data\Islab\Drive\Dataset\FB-Ad\classification\dataset\train-review-5\labels'
    data1 = list(map(lambda x: x.split('.')[0], os.listdir(data1_path)))
    data2 = list(map(lambda x: x.split('.')[0], os.listdir(data2_path)))
    # class_data = list(map(lambda x: x.split('.')[0], os.listdir(class_path)))
    new_path = r'C:\Users\ray93978\Data\Islab\Drive\Dataset\FB-Ad\object-detection\yolo-format\ad-dataset-v2\new_label'
    # same_data = list(map(lambda x: x.split('.')[0], os.listdir(same_path)))

    # temp_path = r'C:\Users\ray93978\Data\Islab\Drive\Dataset\FB-Ad\classification\dataset\train-review-3\same'
    # data1.extend(data2)
    inter_data = get_dataset_intersection(data1, data2)
    print(len(inter_data))
    print(inter_data)
    # print(set(data1).difference(set(data2)))
    ext = 'txt'
    img_ext = 'png'
    # for data in set(data1).difference(set(data2)):
    #     os.remove(os.path.join(data1_path, f'{data}.{ext}'))
    label_names = data_utils.label_names
    target_label = ['其它']
    target_path = r'C:\Users\ray93978\Data\Islab\Ray93978\AD\AdsImgUtils\data\objects\labels'
    for data in tqdm(inter_data):
        # shutil.copy(os.path.join(data2_path, f'{data}.{img_ext}'), os.path.join(target_path, f'{data}.{img_ext}'))
        shutil.copy(os.path.join(data2_path, f'{data}.{ext}'), os.path.join(target_path, f'{data}.{ext}'))
    #     edit_label(os.path.join(data2_path, f'{data}.{ext}'), target_label, label_names)
    #     shutil.move(os.path.join(data2_path, f'{data}.{ext}'), os.path.join(data1_path, f'{data}.{ext}'))


def get_data_by_labels():
    img_dir = r'C:\Users\ray93978\Data\local\label_agreement\paired_file-600\images'
    label_dir = r'C:\Users\ray93978\Data\local\label_agreement\paired_file-600\annotator_1'
    label_names = ['介紹']
    target_dir = r'C:\Users\ray93978\Data\local\label_agreement\paired_file-600\view_ray'
    move_dataset_by_labels(img_dir, label_dir, label_names, target_dir,
                           file_num=5000, img_ext='png', copy=True)


def main4():
    label_name = '教學'
    folder_path = r'C:\Users\ray93978\Data\local\label_survey\train-review-5'
    image_fold = os.path.join(folder_path, label_name, 'images')
    label_fold = os.path.join(folder_path, label_name, 'labels')
    images = get_filenames_by_folder(image_fold)
    labels = get_filenames_by_folder(label_fold)
    target_label = [label_name]
    file_diff = get_dataset_difference(images, labels)
    print(file_diff)
    print(len(file_diff))
    image_paths = list(map(lambda x: os.path.join(r'..\images', f'{x}.png'), file_diff))
    label_paths = list(map(lambda x: os.path.join(label_fold, f'{x}.json'), file_diff))
    create_label_files(image_paths, label_paths, target_label)


def create_missing_label(folder_path):
    for fold in os.listdir(folder_path):
        labels = fold.split(',')
        label_folder = os.path.join(folder_path, fold, 'labels')
        image_folder = os.path.join(folder_path, fold, 'images')
        image_names = get_filenames_by_folder(image_folder)
        label_names = get_filenames_by_folder(label_folder)
        print(labels)
        print(fold)
        # empty_label = get_dataset_difference(label_names, image_names)
        miss_label = get_dataset_difference(image_names, label_names)
        print(get_dataset_difference(label_names, image_names))
        print('miss label:', miss_label)
        print(len(miss_label))
        image_paths = list(map(lambda x: os.path.join(r'..\images', f'{x}.png'), miss_label))
        label_paths = list(map(lambda x: os.path.join(label_folder, f'{x}.json'), miss_label))
        print(image_paths)
        print(label_paths)
        create_label_files(image_paths, label_paths, tar_label=labels)
        print('-' * 10)


def random_move_files(fold, tar_fold, file_num, random=True, copy=True):
    files = os.listdir(fold)
    if random:
        files = shuffle(files, random_state=123)
    files = files[:file_num]
    print(len(files))
    if copy:
        process = shutil.copy
    else:
        process = shutil.move

    for file in tqdm(files):
        process(os.path.join(fold, file), os.path.join(tar_fold, file))


def rgba_to_rgb(img_folder, out_folder):
    for img_path in tqdm(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, img_path))
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(out_folder, img_path), img)


def png_to_jpg(img_folder, output_folder):
    for p in tqdm(list(filter(lambda x: 'png' in x, os.listdir(img_folder)))):
        file_name = p.split('.')[0]
        img = Image.open(os.path.join(img_folder, p))
        img = img.convert('RGB')
        img.save(os.path.join(output_folder, f'{file_name}.jpg'))


def jpg_to_png(img_folder, output_folder):
    for p in tqdm(list(filter(lambda x: 'jpg' in x, os.listdir(img_folder)))):
        file_name = p.split('.')[0]
        img = Image.open(os.path.join(img_folder, p))
        img.save(os.path.join(output_folder, f'{file_name}.png'))


def print_label_distribution(dataset_fold):
    labels = data_utils.load_labels(os.path.join(dataset_fold, 'labels'))
    print(*zip(TARGET_NAME, np.sum(labels, axis=0)), sep='\n')


def jsons_ch_to_en(labels_fold):
    ch_name = data_utils.label_names
    en_name = data_utils.label_names_en
    ch_en_map = {k:v for k, v in zip(ch_name, en_name)}
    for label in tqdm(glob.glob(labels_fold + '/*.json')):
        with open(label, 'r+', encoding='utf-8') as f:
            js = json.load(f)
            flags = js['flags']
            js['flags'] = {ch_en_map[k]: v for k, v in flags.items()}
            f.seek(0)
            f.truncate(0)  # erase file content
            json.dump(js, f, indent=2, ensure_ascii=False)


def ch_label_to_en_label(ch_label, ch_name=data_utils.label_names, en_name=data_utils.label_names_en):
    if isinstance(ch_label, list):
        return list(ch_label_to_en_label(label, ch_name=ch_name, en_name=en_name) for label in ch_label)
    # elif ' ' in ch_label:
    #     return ', '.join(ch_label_to_en_label(ch_label.split(' '), ch_name=ch_name, en_name=en_name))
    elif ',' in ch_label:
        return ','.join(ch_label_to_en_label(ch_label.split(','), ch_name=ch_name, en_name=en_name))
    else:
        return en_name[ch_name.index(ch_label)]


def export_filenames(file_list, tar_path):
    with open(tar_path, 'w+', encoding='utf-8') as f:
        for file in tqdm(file_list):
            f.write(f'{file},\n')


def export_dataset_filenames():
    folder1 = r'C:\Users\ray93978\Data\Islab\Dataset\廣告物件偵測任務\碩論資料集\images\*\*.png'
    folder2 = r'C:\Users\ray93978\Data\Islab\Dataset\廣告圖片分類任務\9類\6500\images'
    file_list1 = get_filenames_by_paths(glob.glob(folder1))
    print(len(file_list1))
    print(file_list1[0])

    file_list2 = get_filenames_by_folder(folder2)
    print(len(file_list2))
    print(file_list2[0])
    export_filenames(file_list1, '../Reports/od_files.txt')
    export_filenames(file_list2, '../Reports/clf_files.txt')


if __name__ == '__main__':
    # main2()
    export_dataset_filenames()

