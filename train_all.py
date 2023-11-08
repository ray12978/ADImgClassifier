import utils.Utils as util
import config as cfg
from config import params
import clearml
from clearml import Task


def main():
    # clearml 初始化
    clearml.browser_login()
    task = Task.init(
        project_name=f"AD Classification",
        task_name=cfg.exp_name,
        output_uri=True
    )
    task.connect(params)

    # 訓練流程
    exp_process = util.ExperimentProcess(
        train_mode='DLR',  # 組態設定
        save_model=cfg.save_model,
        exp_name=cfg.exp_name,
        data_size=cfg.data_size
    )
    exp_process.train_models()
    exp_process.train_mode = 'baseline'
    exp_process.train_models()


if __name__ == '__main__':
    main()
