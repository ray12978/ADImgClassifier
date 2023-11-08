import torch
import torch.nn as nn
from torchvision import transforms

import config as cfg
import utils.data_utils as data_utils
import utils.Utils as util
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


transformer_serial_names = [
    'BeitForImageClassification',
    'SegformerForImageClassification',
    'ViTForImageClassification',
    'ConvNextForImageClassification',
    'ConvNextV2ForImageClassification',
    'ImageGPTForImageClassification',
    'SwinForImageClassification',
    'Swinv2ForImageClassification',
    'EfficientNetForImageClassification'
]


timm_serial_names = [
    'Beit'
]


P6_yolo_models = [
    'yolov7e6e',
]


def init_model(model_name, pretrained=True, pretrained_yolo=cfg.pretrained_yolo, output_num=data_utils.class_num):
    # 初始化模型，並回傳修改輸出層後的模型
    util.set_torch_seed()
    print('Initialing...', model_name)
    processor = None
    if 'yolo' in model_name:
        if pretrained_yolo:
            model = torch.load(f'models/trained_yolo/{model_name}.pt')['model'].float()
        else:
            model = torch.load(f'models/yolo/{model_name}.pt')['model'].float()
        # image_size = 640 if model_name not in P6_yolo_models else 1280
        image_size = 640  # 1280 size 需求記憶體大於128GB
        processor = data_utils.get_yolo_transform(image_size)
    elif 'timm/' in model_name:
        model = timm.create_model(model_name, pretrained=pretrained)
        processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif '/' in model_name:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
    return init_layer(model, model_name, output_num=output_num), processor


def init_models(model_names, pretrained=True, pretrained_yolo=cfg.pretrained_yolo):
    items = [init_model(model_name, pretrained, pretrained_yolo=pretrained_yolo) for model_name in model_names]
    return [item[0] for item in items], [item[1] for item in items]


def init_layer(model, model_name, output_num=data_utils.class_num):
    # 修改模型輸出層輸出數量
    if 'yolov5' in model_name:
        model = yolov5_edit(model)
    elif 'yolov6' in model_name:
        model = yolov6_edit(model)
    elif 'yolov7' in model_name:
        model = yolov7_edit(model)
    elif 'yolov8' in model_name:
        model = yolov8_edit(model)
    elif model.__class__.__name__ == 'GoogLeNet' or model.__class__.__name__ == 'ResNet':
        model.fc = nn.Linear(model.fc.in_features, output_num)
    elif model.__class__.__name__ == 'ImageGPTForImageClassification':
        model.score = nn.Linear(model.score.in_features, output_num)
    elif model.__class__.__name__ in transformer_serial_names:
        model.classifier = nn.Linear(model.classifier.in_features, output_num)
    elif model.__class__.__name__ in timm_serial_names:
        model.head = nn.Linear(model.head.in_features, output_num)
    else:
        model.classifier[-1] = nn.Linear(4096, output_num)
    return model


def load_model(model_name, model_path, state_dict=True, output_num=data_utils.class_num):
    # 讀取模型權重
    model, processor = init_model(model_name, pretrained=True, output_num=output_num)
    if state_dict:
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.load(model_path)
    if type(model) == dict:
        model = model['model'].float()
    model.eval()
    return model, processor


def load_models(models, models_paths, state_dict=True, output_num=data_utils.class_num):
    clfs = []
    names = []
    pbar = tqdm(zip(models, models_paths), total=len(models_paths))
    for md, mp in pbar:
        pbar.set_description(f'loading... {mp}, path: {mp}')
        model, name = load_model(md, mp, state_dict=state_dict, output_num=output_num)
        model.eval()
        clfs.append(model)
        names.append(name)
    return clfs, names


class Classify(nn.Module):
    # YOLOC模型使用的分類器
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


def get_module_idx_by_name(model, module_name):
    # 根據模型架構名稱取得索引值
    module_idxes = list(map(lambda x: x[0], model.model.named_modules()))
    module_names = list(map(lambda x: x[1].__class__.__name__, model.model.named_modules()))
    idx = module_names.index(module_name)
    return int(module_idxes[idx])


# 修改YOLO系列模型架構，轉換成"YOLOC"
def yolov5_edit(model, nc=data_utils.class_num):
    idx = get_module_idx_by_name(model, 'SPPF')
    model.model = model.model[:idx+1]
    m = model.model[-1]
    ch = m.cv2.conv.out_channels
    c = Classify(ch, nc)
    c.i, c.f = m.i, m.f
    model.model.add_module('classifier', c)
    for p in model.parameters():
        p.requires_grad = True
    return model


def yolov6_edit(model, nc=data_utils.class_num):
    model = model.backbone
    if list(model.ERBlock_5[-1].named_modules())[1][0] == 'sppf':
        model.ERBlock_5[-1] = model.ERBlock_5[-1].sppf
        model.ERBlock_5[-1].cv1 = model.ERBlock_5[-1].cv1.block
        model.ERBlock_5[-1].cv2 = model.ERBlock_5[-1].cv2.block
    m = model.ERBlock_5[-1]
    ch = m.cv2.bn.num_features
    c = Classify(ch, nc)
    model.ERBlock_5.add_module('classifier', c)
    for p in model.parameters():
        p.requires_grad = True
    return model


def yolov7_edit(model, nc=data_utils.class_num):
    idx = get_module_idx_by_name(model, 'SPPCSPC')
    model.model = model.model[:idx+1]
    m = model.model[-1]
    ch = m.cv7.conv.out_channels
    c = Classify(ch, nc)
    c.i, c.f = m.i, m.f
    model.model.add_module('classifier', c)
    for p in model.parameters():
        p.requires_grad = True
    return model


def yolov8_edit(model, nc=data_utils.class_num):
    idx = get_module_idx_by_name(model, 'SPPF')
    model.model = model.model[:idx+1]
    m = model.model[-1]
    ch = m.cv2.conv.out_channels
    c = Classify(ch, nc)
    c.i, c.f = m.i, m.f
    model.model.add_module('classifier', c)
    for p in model.parameters():
        p.requires_grad = True
    return model


def init_diff_optimizer(model, model_name, opti):
    # 優化器設定差分學習率
    if model.__class__.__name__ == 'GoogLeNet':
        optimizer = opti([{'params': model.conv1.parameters(), 'lr': cfg.first_lr},
                          {'params': model.conv2.parameters(), 'lr': cfg.first_lr},
                          {'params': model.conv3.parameters(), 'lr': cfg.second_lr},
                          {'params': model.fc.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'ResNet':
        optimizer = opti([{'params': model.layer1.parameters(), 'lr': cfg.first_lr},
                          {'params': model.layer2.parameters(), 'lr': cfg.first_lr},
                          {'params': model.layer3.parameters(), 'lr': cfg.first_lr},
                          {'params': model.layer4.parameters(), 'lr': cfg.second_lr},
                          {'params': model.fc.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'BeitForImageClassification':
        optimizer = opti([{'params': model.beit.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.beit.encoder.layer[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.beit.encoder.layer[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.beit.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.beit.pooler.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'SegformerForImageClassification':
        optimizer = opti([{'params': model.segformer.encoder.patch_embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.segformer.encoder.block[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.segformer.encoder.block[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'ViTForImageClassification':
        optimizer = opti([{'params': model.vit.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.vit.encoder.layer[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.vit.encoder.layer[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.vit.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'ImageGPTForImageClassification':
        optimizer = opti([{'params': model.transformer.wte.parameters(), 'lr': cfg.first_lr},
                          {'params': model.transformer.wpe.parameters(), 'lr': cfg.first_lr},
                          {'params': model.transformer.h[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.transformer.h[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.transformer.ln_f.parameters(), 'lr': cfg.last_lr},
                          {'params': model.score.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'SwinForImageClassification':
        optimizer = opti([{'params': model.swin.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.swin.encoder.layers[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.swin.encoder.layers[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.swin.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'Swinv2ForImageClassification':
        optimizer = opti([{'params': model.swinv2.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.swinv2.encoder.layers[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.swinv2.encoder.layers[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.swinv2.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'ConvNextForImageClassification':
        optimizer = opti([{'params': model.convnext.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.convnext.encoder.stages[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.convnext.encoder.stages[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.convnext.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'ConvNextV2ForImageClassification':
        optimizer = opti([{'params': model.convnextv2.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.convnextv2.encoder.stages[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.convnextv2.encoder.stages[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.convnextv2.layernorm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'Beit':
        optimizer = opti([{'params': model.patch_embed.parameters(), 'lr': cfg.first_lr},
                          {'params': model.blocks[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.blocks[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.norm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.fc_norm.parameters(), 'lr': cfg.last_lr},
                          {'params': model.head.parameters(), 'lr': cfg.last_lr}])
    elif model.__class__.__name__ == 'EfficientNetForImageClassification':
        optimizer = opti([{'params': model.efficientnet.embeddings.parameters(), 'lr': cfg.first_lr},
                          {'params': model.efficientnet.encoder.blocks[0:-1].parameters(), 'lr': cfg.first_lr},
                          {'params': model.efficientnet.encoder.blocks[-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.efficientnet.encoder.top_conv.parameters(), 'lr': cfg.last_lr},
                          {'params': model.efficientnet.encoder.top_bn.parameters(), 'lr': cfg.last_lr},
                          {'params': model.efficientnet.pooler.parameters(), 'lr': cfg.last_lr},
                          {'params': model.classifier.parameters(), 'lr': cfg.last_lr}])
    elif 'yolov6' in model_name:
        optimizer = opti([{'params': model.stem.parameters(), 'lr': cfg.first_lr},
                          {'params': model.ERBlock_2.parameters(), 'lr': cfg.first_lr},
                          {'params': model.ERBlock_3.parameters(), 'lr': cfg.first_lr},
                          {'params': model.ERBlock_4.parameters(), 'lr': cfg.first_lr},
                          {'params': model.ERBlock_5[:-2].parameters(), 'lr': cfg.first_lr},
                          {'params': model.ERBlock_5[-2].parameters(), 'lr': cfg.second_lr},
                          {'params': model.ERBlock_5[-1].parameters(), 'lr': cfg.last_lr}])
    elif 'yolo' in model_name:
        optimizer = opti([{'params': model.model[0:-2].parameters(), 'lr': cfg.first_lr},
                          {'params': model.model[-2].parameters(), 'lr': cfg.second_lr},
                          {'params': model.model[-1].parameters(), 'lr': cfg.last_lr}])
    else:
        optimizer = opti([{'params': model.features.parameters(), 'lr': cfg.first_lr},
                          {'params': model.classifier[0:-1].parameters(), 'lr': cfg.second_lr},
                          {'params': model.classifier[-1].parameters(), 'lr': cfg.last_lr}])
    return optimizer


def init_baseline_optimizer(model, opti=cfg.optimizer):
    # 優化器設定基準組態
    return opti(model.parameters(), lr=cfg.second_lr)


def init_optimizer(model, model_name, opti=cfg.optimizer, train_mode='baseline'):
    if train_mode == 'baseline':
        return init_baseline_optimizer(model, opti)
    elif train_mode == 'DLR':
        return init_diff_optimizer(model, model_name, opti)


def init_optimizers(models, model_names, opti=cfg.optimizer, train_mode=''):
    return [init_optimizer(model, model_name, opti, train_mode) for model, model_name in zip(models, model_names)]


def count_parameters(model, grad=True):
    # 計算模型參數量
    if grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
