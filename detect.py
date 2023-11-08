import torch
from torch.utils.data import DataLoader
from utils import data_utils, model_utils
from utils import Utils
import os


is_cuda = torch.cuda.is_available()
model_name = 'yolov7'  # 模型名稱
model_path = 'Results/transformer_model/YOLO/yolov7/differential.pt'  # 模型權重路徑
detect_path = 'object detection/data'  # 主資料夾路徑
image_path = os.path.join(detect_path, 'images')  # 圖片資料夾路徑
label_path = os.path.join(detect_path, 'labels')  # 標籤輸出資料夾路徑


def main():
    model, processor = model_utils.load_model(model_name, model_path, state_dict=False)  # 讀取模型
    if is_cuda:
        model.cuda()
    images, _, paths = data_utils.load_data_by_folder(image_path, None, need_path=True, ext='png', has_label=False)
    dataset, _ = data_utils.get_transformed_image(images, processor)
    data_loader = DataLoader(dataset, batch_size=64)
    predicts = Utils.detect(model, data_loader, filter_=True)
    Utils.pre_label(predicts, paths, label_path)


if __name__ == '__main__':
    main()
