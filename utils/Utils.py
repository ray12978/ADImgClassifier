import copy
import gc
import json
import os
import platform
import random
import re
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm

import config as cfg
import utils.data_utils as data_utils
import utils.model_utils as model_utils
from config import params
from trainer import Trainer
from utils import get_logger, get_result_subdir

is_cuda = True  # torch.cuda.is_available()
device = torch.device('cuda')  # if torch.cuda.is_available() else torch.device('cpu')


class ExperimentProcess:
    def __init__(
            self, model_names=cfg.model_names, epochs=cfg.num_epochs,
            train_mode='', data_size=None, save_model=True, exp_name=cfg.exp_name
    ):
        self.epochs = epochs
        self.model_names = model_names
        self.optimizers = []
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_mode = train_mode
        self.result_dirs, self.exp_dir = get_result_subdir(self.model_names, exp_name=exp_name)
        self.log_path = os.path.join(self.exp_dir, 'process_log.txt')
        self.optimizer_name = None
        self.data_size = data_size
        self.save_model = save_model
        self.logger = get_logger(fp=self.exp_dir)
        self.logger.info('hyp params:' + ', '.join(f'{k}={v}' for k, v in params.items()))
        self.trainer = None

    def train_models(self):
        for i, model_name in enumerate(self.model_names):
            os.makedirs(self.result_dirs[i], exist_ok=True)
            self.trainer = Trainer(model_name, self.train_mode, data_size=self.data_size,
                                   save_model=self.save_model, output_dir=self.result_dirs[i],
                                   logger=self.logger, log_path=self.log_path)
            self.trainer.run_train()
            del self.trainer
            gc.collect()


def fit(model, data_loader, optimizer, loss_fn, model_name='', nb=0):
    model.train()
    running_loss = 0.0
    for i, (data, target) in data_loader:
        if is_cuda:
            model.cuda()
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model_output(model, data)
        if 'yolo' in model_name:
            with amp.autocast(enabled=False):
                loss = loss_fn(output, target)
            running_loss += loss.item()
        else:
            loss = loss_fn(output, target)
            running_loss += loss_fn(output, target, reduction='sum').item()
        loss.backward()
        optimizer.step()
    return running_loss / nb


def detect(model, data, filter_=True):
    model.eval()
    preds = []
    for data in tqdm(data):
        if is_cuda:
            model.cuda()
            data = data.cuda()
        output = model_output(model, data)
        for i, example in enumerate(torch.sigmoid(output)):
            if filter_:
                pred = predict_filter(example)
            else:
                pred = []
                for p in example:
                    pred.append(1) if p > 0.5 else pred.append(0)
            preds.append(pred)
    return preds


def early_stop_train(epochs, model, optimizer,
                     loss_fn, t_loader, v_loader,
                     min_delta=0.1, patience=10):
    train_losses = []
    val_losses = []
    total_time = 0
    best_loss = None
    counter = 0
    early_stop = False
    for epoch in range(1, epochs + 1):
        start = time.time()
        epoch_loss = fit(model, t_loader, optimizer, loss_fn)
        train_losses.append(epoch_loss)
        end = time.time()
        total_time += end - start

        _, val_loss, report = predict(model, v_loader, loss_fn)
        val_losses.append(val_loss)
        if epoch % 5 == 0:
            print(f'Epoch {epoch}/{epochs}')
            print(f'Training loss is {epoch_loss:5.2f}')
            print(f'Validation loss is {val_loss:5.2f}/Micro-avg. F-score is {report:5.3f}')
            print('-' * 50)

        if best_loss is None:
            best_loss = val_loss
        elif best_loss - val_loss > min_delta:
            best_loss = val_loss
            counter = 0
        elif best_loss - val_loss < min_delta:
            counter += 1
            print(f'INFO: Early stopping counter {counter} of {patience}')
            if counter >= patience:
                print('INFO: Early stopping')
                early_stop = True
        if early_stop:
            break
    print(f'Finished within {total_time:.3f} seconds.')
    torch.cuda.empty_cache()
    return train_losses, val_losses


def model_output(model, data):
    if model.__class__.__name__ in model_utils.transformer_serial_names:
        output = model(data).logits
    else:
        output = model(data)
    if type(output) == tuple:
        output = output[-1]
    return output


def predict_filter(probs, threshold=cfg.THRESHOLD):
    probs = list(probs)
    label_map = {0: None, 1: (2, 3, 4), 5: (6,), 7: None, 8: None}
    main_labels = list(label_map.keys())
    sub_labels = list(label_map.values())
    main_probs = list(v for i, v in enumerate(probs) if i in main_labels)
    max_main = probs.index(max(main_probs))
    res = []
    probs[max_main] = 1

    for mlab, slab in zip(main_labels, sub_labels):
        res.append(1 if probs[mlab] == 1 else 0)
        if not slab:
            continue
        for sub in slab:
            res.append(1 if probs[sub] > threshold and probs[mlab] == 1 else 0)
    return res


def pre_label(predicts, filepaths, output_folder):
    separator = '\\' if platform.system() == 'Windows' else '/'
    js_format = cfg.json_format_path
    with open(js_format, encoding='utf-8') as f:
        text = json.load(f)
    print('pre-labeling images...')
    for pred, fp in tqdm(zip(predicts, filepaths), total=len(predicts)):
        filename = fp.split(separator)[-1].split('.')[0]
        file = fp.split(separator)[-1]
        file_dir = fp.split(separator)[-2]
        pred_json = copy.deepcopy(text)
        true_list = [i for i, p in enumerate(pred) if p == 1]
        pred_json['imagePath'] = join('..', file_dir, file)
        for true in list(true_list):
            pred_json['flags'][data_utils.label_names[true]] = True
        os.makedirs(output_folder, exist_ok=True)
        with open(join(output_folder, f'{filename}.json'), 'w', encoding='utf-8') as f:
            json.dump(pred_json, f, ensure_ascii=False, indent=2)
    print(f'label save at: {output_folder}')


def save_reports(text, path):
    text_list = [list(filter(None, re.split('\s{2,}', t))) for t in text.split('\n')]
    text_list = (list(filter(None, text_list)))
    text_list[0].insert(0, 'labels')
    with open(path, 'w') as fw:
        for text in text_list:
            fw.writelines(','.join(text) + '\n')
    print(f'wrote {path}')


def save_losses(losses, path):
    with open(path, 'w') as fw:
        for i, loss in enumerate(losses):
            if i != len(losses) - 1:
                fw.writelines(f'{loss},')
            else:
                fw.writelines(f'{loss}')
    print(f'wrote {path}')


def draw_plots(x: list, y, x_label='training loss', y_label='validation loss', save_path=None):
    plt.plot(range(1, len(x) + 1), x, label=x_label)
    if y is not None:
        plt.plot(range(1, len(y) + 1), y, label=y_label)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.clf()


def write_log(log_path, content):
    with open(log_path, 'a', encoding='utf8') as f:
        if isinstance(content, str):
            f.writelines('{}\n'.format(content))
        else:
            f.writelines('{}\n'.format('\n'.join(content)))


def one_hot_to_label_name(encode, target_name):
    if isinstance(encode[0], list):
        return list(one_hot_to_label_name(c, target_name) for c in encode)
    else:
        return list(target for i, target in enumerate(target_name) if encode[i] == 1)


def set_torch_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
