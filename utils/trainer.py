import datetime
import os
import time

import torch
from sklearn.metrics import classification_report, f1_score
from torch.cuda import amp
from tqdm import tqdm

import config as cfg
from utils import data_utils, model_utils
from utils.torch_utils import EarlyStopping
from utils.Utils import draw_plots, model_output, predict_filter, save_losses, save_reports, set_torch_seed, write_log

GB_SIZE = 1073741824


class Trainer:
    def __init__(self,
                 model_name,
                 train_mode,
                 data_size=None,
                 save_model=True,
                 output_dir='',
                 logger=None,
                 log_path=''):
        self.epochs = cfg.num_epochs
        self.loss_fn = cfg.loss_fn
        self.train_losses = []
        self.val_losses = []
        self.f_scores = []
        self.total_time = 0
        self.start_time = time.time()
        self.total_mem = '%.3gG' % (torch.cuda.mem_get_info()[1] / GB_SIZE if torch.cuda.is_available() else 0)
        self.model_name = model_name
        self.train_mode = train_mode
        self.data_size = data_size
        self.save_model = save_model
        self.output_dir = output_dir
        self.is_cuda = torch.cuda.is_available()
        self.logger = logger
        self.log_path = log_path
        self.patience = cfg.patience
        self.fitness = None
        self.best_fitness = None

    def run_train(self):
        self._setup_train()
        self.train()
        if self.test_loader:
            report = self.validate(self.test_loader, report='')
            self.logger.info(report)
            write_log(self.log_path, report)
            save_reports(report, os.path.join(self.output_dir, f'{self.train_mode}.csv'))
        self.save_all_losses()
        self.draw_all_plots()
        self.save_hyper_params()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            pbar = enumerate(self.train_loader)
            num_batch = len(self.train_loader)
            self.model.train()
            self.logger.info(('\n' + '%10s' * 5) % ('Epoch', 'config', 'model', 'gpu_mem', 't_loss'))
            pbar = tqdm(pbar, total=num_batch)
            start = time.time()

            running_loss = 0.0
            running_time = 0
            batch_start = time.time()
            for i, (data, target) in pbar:
                if self.is_cuda:
                    data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = model_output(self.model, data)
                if 'yolo' in self.model_name:
                    with amp.autocast(enabled=False):
                        loss = self.loss_fn(output, target)
                    running_loss += loss.item()
                else:
                    loss = self.loss_fn(output, target)
                    running_loss += self.loss_fn(output, target, reduction='sum').item()
                loss.backward()
                self.optimizer.step()
                running_time += time.time() - batch_start

                mem = '%.3gG' % (torch.cuda.memory_reserved() / GB_SIZE if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 4 + '%10.4g' * 1) % \
                    ('%g/%g' % (epoch, self.epochs), self.train_mode, self.model_name, '%s/%s' % (mem, self.total_mem),
                     running_loss / (i + 1))
                pbar.set_description(s)
            epoch_loss = running_loss / num_batch
            self.train_losses.append(epoch_loss)
            self.total_time += time.time() - start
            if self.valid_loader:
                _, val_loss, report = self.validate(self.valid_loader)
                self.fitness = val_loss

                self.val_losses.append(val_loss)
                self.f_scores.append(report)
                self.logger.info(('\n' + '%10s' * 3) % ('v_loss', 'micro-f1', '\tspend time') +
                                 ('\n' + '%10.4g' * 2) % (val_loss, report) +
                                 f'    {datetime.timedelta(seconds=(time.time() - self.start_time))}')
                self.stop = self.stopper(epoch + 1, 1 - val_loss)
                if self.save_model:
                    self._save_model()

            self.logger.info(('\n' + '%10s' * 1) % '\tspend time' +
                             f'\t{datetime.timedelta(seconds=(time.time() - self.start_time))}')
            if self.stop:
                self.logger.info(f'Early stopped at epoch[{epoch}]. '
                                 f'To disable early stop, set patience to 0. i.e. patience=0')
                break
        self.logger.info(f'Finished within {datetime.timedelta(seconds=self.total_time)}.')
        torch.cuda.empty_cache()
        return self.model, self.train_losses, self.val_losses, self.f_scores

    def validate(self, dataset, report='f1', major_label_only=False, pred_filter=cfg.PREDICT_FILTER):
        self.model.eval()
        running_loss = 0.0
        all_target = []
        all_predict = []
        for data, target in dataset:
            all_target.extend(target.cpu().tolist())
            if self.is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model_output(self.model, data)
            running_loss += self.loss_fn(output, target, reduction='sum').item()
            predicts = []
            for probs in torch.sigmoid(output):
                if pred_filter:
                    pred = predict_filter(probs)
                else:
                    pred = []
                    for p in probs:
                        pred.append(1) if p > cfg.THRESHOLD else pred.append(0)
                predicts.append(pred)
            all_predict.extend(predicts)
        if major_label_only:
            target_names = data_utils.major_label_names
            major_index = [i for i, v in enumerate(self.test_loader.dataset.classes) if v in target_names]
            all_predict = list(map(lambda x: [v for i, v in enumerate(x) if i in major_index], all_predict))
            all_target = list(map(lambda x: [v for i, v in enumerate(x) if i in major_index], all_target))
        else:
            target_names = self.test_loader.dataset.classes

        if report == 'f1':
            reports = f1_score(all_target, all_predict, average='micro')
        else:
            reports = classification_report(all_target, all_predict, target_names=target_names,
                                            zero_division=0, digits=6)
        loss = running_loss / len(self.test_loader.dataset)
        if not self.best_fitness or self.best_fitness > loss:
            self.best_fitness = loss
        return all_predict, loss, reports

    def _setup_train(self):
        set_torch_seed()
        self.model, self.processor = model_utils.init_model(self.model_name)
        self.optimizer = model_utils.init_optimizer(self.model, self.model_name, train_mode=self.train_mode)
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataset()
        self.stopper, self.stop = EarlyStopping(patience=cfg.patience), False
        if self.is_cuda:
            self.model.cuda()

    def get_dataset(self):
        data_loaders = data_utils.get_dataloader(transform=self.processor, data_size=self.data_size)
        return data_loaders[0], data_loaders[1], data_loaders[2]

    def _save_model(self):
        torch.save(self.model, os.path.join(self.output_dir, f'{self.train_mode}_last.pt'))
        if self.best_fitness == self.fitness:
            torch.save(self.model, os.path.join(self.output_dir, f'{self.train_mode}_best.pt'))


    def save_all_losses(self):
        save_losses(self.train_losses, os.path.join(self.output_dir, f'{self.train_mode}_train_losses.csv'))
        save_losses(self.val_losses, os.path.join(self.output_dir, f'{self.train_mode}_valid_losses.csv'))
        save_losses(self.f_scores, os.path.join(self.output_dir, f'{self.train_mode}_f1_scores.csv'))

    def draw_all_plots(self):
        draw_plots(self.train_losses, self.val_losses,
                   save_path=os.path.join(self.output_dir, f'{self.train_mode}_losses.png'))
        draw_plots(self.f_scores, None, x_label='f1-scores',
                   save_path=os.path.join(self.output_dir, f'{self.train_mode}_f1_scores.png'))

    def save_hyper_params(self):
        with open(os.path.join(self.output_dir, 'hyper_params.csv'), 'w') as f:
            f.writelines('{},{}\n'.format('model', self.model_name))
            f.writelines('{},{}\n'.format('train batch size', cfg.train_batch))
            f.writelines('{},{}\n'.format('valid batch size', cfg.valid_batch))
            f.writelines('{},{}\n'.format('test batch size', cfg.test_batch))
            f.writelines('{},{}\n'.format('epochs', self.epochs))
            f.writelines('{},{}\n'.format('loss function', cfg.loss_fn.__name__))
            f.writelines('{},{}\n'.format('optimizer', cfg.optimizer.__name__))
            f.writelines('{},{}\n'.format('first lr', cfg.first_lr))
            f.writelines('{},{}\n'.format('second lr', cfg.second_lr))
            f.writelines('{},{}\n'.format('last lr', cfg.last_lr))
            f.writelines('{},{}\n'.format('threshold', cfg.THRESHOLD))
            f.writelines('{},{}\n'.format('random seed', cfg.SEED))
            f.writelines('{},{}\n'.format('predict filter', cfg.PREDICT_FILTER))
