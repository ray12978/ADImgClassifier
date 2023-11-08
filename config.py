import os
from torch.optim import SGD, Adam
import torch.nn.functional as F

# file path
dataset_path = r''  # 資料夾路徑
json_format_path = 'Data/json_format.json'
flag_path = 'Data/flags.txt'

exp_name = 'TEST'  # 實驗名稱
save_model = False
data_size = None  # 資料大小限制，None表示不限制
pretrained_yolo = True


# hyperparameter
train_batch = 64
valid_batch = 24
test_batch = 16


first_lr = 1e-6
second_lr = 1e-4
last_lr = 1e-2
num_epochs = 400
SEED = 123
THRESHOLD = 0.5
optimizer = SGD
loss_fn = F.binary_cross_entropy_with_logits
PREDICT_FILTER = True
patience = 50

cnn_base_model = [
    'alexnet', 'googlenet', 'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152', 'vgg11',
    'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn',
    'vgg16_bn', 'vgg19_bn'
]

yoloc_model = [
    'yolov7x',
    'yolov7',
    'yolov7e6e',
    'yolov8m',
    'yolov8l',
    'yolov8x',
]


paper_cnn_model = [
    'alexnet', 'googlenet',
    'resnet101', 'vgg13'
]

microsoft_model = [
    'microsoft/beit-base-patch16-224-pt22k-ft22k',
    'microsoft/swin-base-patch4-window7-224-in22k',
    'microsoft/swin-large-patch4-window7-224-in22k',
    'hf-hub:timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k',
    'microsoft/swinv2-base-patch4-window8-256',
]

microsoft_model_large = [
    'microsoft/beit-large-patch16-224-pt22k-ft22k',
    'hf-hub:timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k',
    'microsoft/swinv2-large-patch4-window12-192-22k',
]

google_model = [
    'google/vit-base-patch16-224',
    'google/efficientnet-b2',
    'google/efficientnet-b3',
    'google/efficientnet-b1',
    'google/efficientnet-b0',
]

google_model_large = [
    'google/vit-large-patch16-224',
]

google_model_xlarge = [
    'google/efficientnet-b4',
    'google/efficientnet-b5',
    'google/efficientnet-b6',
    'google/efficientnet-b7',
]

nvidia_model = [
    'nvidia/mit-b0',
]

nvidia_model_large = [
    'nvidia/mit-b1',
]

nvidia_model_xlarge = [
    'nvidia/mit-b2',
    'nvidia/mit-b3',
    'nvidia/mit-b4',
    'nvidia/mit-b5',
]

openai_model = [
    'openai/imagegpt-small',
    'openai/imagegpt-medium',
]

facebook_model = [
    'facebook/convnext-base-224-22k-1k',
    'facebook/convnextv2-nano-22k-224',
    'facebook/convnextv2-tiny-22k-224',
    'facebook/convnextv2-base-22k-224',
]

facebook_model_large = [
    'facebook/convnext-large-224-22k-1k',
]

facebook_model_xlarge = [
    'facebook/convnextv2-large-22k-224',
    'facebook/convnext-xlarge-224-22k-1k',
]

base_transformers_model = [
    *microsoft_model,
    *google_model,
    *nvidia_model,
    *facebook_model
]

transformers_model_large = [
    *microsoft_model_large,
    *google_model_large,
    *nvidia_model_large,
    *facebook_model_large,
]

transformers_model_xlarge = [
    *openai_model,
    *nvidia_model_xlarge,
    *google_model_xlarge,
    *facebook_model_xlarge
]

all_model = [
    *base_transformers_model,
    *transformers_model_large,
    *transformers_model_xlarge
]

model_names = base_transformers_model


# hyper parameters dict
params = {
    'train_batch': train_batch,
    'valid_batch': valid_batch,
    'test_batch': test_batch,
    'first_lr': first_lr,
    'second_lr': second_lr,
    'last_lr': last_lr,
    'num_epochs': num_epochs,
    'SEED': SEED,
    'THRESHOLD': THRESHOLD,
    'optimizer': optimizer,
    'loss_fn': loss_fn,
    'PREDICT_FILTER': PREDICT_FILTER,
    'model_names': model_names,
    'pretrained_yolo': pretrained_yolo
}
"""



[Vision Transformer (ViT)]
vit_base_patch16_224

[PiT]
pit_ti_distilled_224
pit_xs_distilled_224
pit_s_distilled_224
pit_b_distilled_224

[beit]
beit_large_patch16_224_in22k

[resnet]
resnet18', 
'resnet34', 
'resnet50', 
'resnet101', 
'resnet152'

[ConvNeXt]

[VGGNET]
'vgg11', 
'vgg13',
'vgg16', 
'vgg19', 
'vgg11_bn', 
'vgg13_bn', 
'vgg16_bn', 
'vgg19_bn'

[YOLO]
'yolov5s', 
'yolov5x'

[Other]
"""
