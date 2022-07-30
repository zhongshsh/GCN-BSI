#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle  # 设置进程名模块
import dataset
from model import BGCN, BGCN_Info
from utils import check_overfitting, early_stop, logger
from train import train
from metric import Recall, NDCG, MRR
from config import CONFIG
from test import test
import loss
from itertools import product
import time

#  from utils.visshow import VisShow
from tensorboardX import SummaryWriter


def main():
    #  set env
    setproctitle.setproctitle(f"train{CONFIG['name']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_id"]
    device = torch.device("cuda")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # torch.device代表将torch.Tensor分配到的设备的对象。
    # torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号。
    # 如果设备序号不存在，则为当前设备。如：torch.Tensor用设备构建‘cuda’的结果等同于‘cuda：X’，
    # 其中X是torch.cuda.current_device()的结果。

    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load data 数据预处理，处理类在dataset.py，主要使用coo
    # 包括是否是hard训练的判断
    # 得到的data是自定义类
    # ub_train,ub_tune,ui,bi
    # 调换成：ui_train,ui_tune,ub,bi
    bundle_train_data, bundle_test_data, item_data, assist_data = dataset.get_dataset(
        CONFIG["path"], CONFIG["dataset_name"], task=CONFIG["task"]
    )

    # 用于train，DataLoader定义的都是一些调用方法，无处理函数
    # 该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。
    train_loader = DataLoader(
        bundle_train_data, 1024, True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        bundle_test_data, 4096, False, num_workers=16, pin_memory=True
    )

    #  pretrain
    if "pretrain" in CONFIG:
        pretrain = torch.load(CONFIG["pretrain"], map_location="cpu")
        print("load pretrain")

    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b  # ground_truth_u_b -> sp.coo_matrix()
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    #  metric 效果评估
    metrics = [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
    TARGET = "Recall@20"

    #  loss
    loss_func = loss.BPRLoss("mean")

    #  log
    log = logger.Logger(
        os.path.join(
            CONFIG["log"],
            CONFIG["dataset_name"],
            f"{CONFIG['model']}_{CONFIG['task']}",
            "",
        ),
        "best",
        checkpoint_target=TARGET,
    )

    theta = 0.6

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

    for lr, decay, message_dropout, node_dropout in product(
        CONFIG["lrs"],
        CONFIG["decays"],
        CONFIG["message_dropouts"],
        CONFIG["node_dropouts"],
    ):
        # vis = VisShow('localhost', 16666,
        #               f'{CONFIG['dataset_name']}-{MODELTYPE.__name__}-{decay}-{lr}-{theta}-3layer')

        visual_path = os.path.join(
            CONFIG["visual"],
            CONFIG["dataset_name"],
            f"{CONFIG['model']}_{CONFIG['task']}",
            f"{time_path}@{CONFIG['note']}",
            f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}",
        )

        # model
        if CONFIG["model"] == "bundle_info":
            graph = [ub_graph, ui_graph, bi_graph]
            info = BGCN_Info(64, decay, message_dropout, node_dropout, 1)
            model = BGCN(info, assist_data, graph, device, pretrain=None)
            model.to(device)

        assert model.__class__.__name__ == CONFIG["model"]

        # op
        op = optim.Adam(model.parameters(), lr=lr)
        # env
        env = {
            "lr": lr,
            "op": str(op).split(" ")[0],  # Adam
            "dataset": CONFIG["dataset_name"],
            "model": CONFIG["model"],
            "sample": CONFIG["sample"],
        }
        #  print(info)

        #  continue training
        if CONFIG["sample"] == "hard" and "conti_train" in CONFIG:
            model.load_state_dict(torch.load(CONFIG["conti_train"]))
            print("load model and continue training")

        retry = CONFIG["retry"]  # =1
        while retry >= 0:  # 用于排错
            # log
            log.update_modelinfo(info, env, metrics)
            # try:
            # train & test
            early = CONFIG["early"]
            train_writer = SummaryWriter(log_dir=visual_path, comment="train")
            test_writer = SummaryWriter(log_dir=visual_path, comment="test")
            for epoch in range(CONFIG["epochs"]):
                # train
                # test process
                trainloss = train(
                    model, epoch + 1, train_loader, op, device, CONFIG, loss_func
                )
                train_writer.add_scalars("loss/single", {"loss": trainloss}, epoch)
                # vis.update('train loss', [epoch], [trainloss])

                # test
                if epoch % CONFIG["test_interval"] == 0:
                    # 此处原代码出错
                    # output_metrics = test(model, epoch+1, test_loader, device, CONFIG, metrics)
                    output_metrics = test(model, test_loader, device, CONFIG, metrics)

                    for metric in output_metrics:
                        test_writer.add_scalars(
                            "metric/all", {metric.get_title(): metric.metric}, epoch
                        )
                        if metric == output_metrics[0]:
                            test_writer.add_scalars(
                                "metric/single",
                                {metric.get_title(): metric.metric},
                                epoch,
                            )

                    # log
                    log.update_log(metrics, model)
                    #  # show(log.metrics_log)

                    # check overfitting
                    if epoch > 10:
                        if check_overfitting(log.metrics_log, TARGET, 1, show=False):
                            break
                    # early stop
                    early = early_stop(log.metrics_log[TARGET], early, threshold=0)
                    if early <= 0:
                        break
                train_writer.close()
                test_writer.close()

                log.close_log(TARGET)
                retry = -1
            # except Exception as e:
            #     retry -= 1

    log.close()


if __name__ == "__main__":
    main()
