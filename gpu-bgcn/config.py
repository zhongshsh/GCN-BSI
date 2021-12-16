#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONFIG = {
    'name': '@bgcn',
    'path': '../data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0,1",
    'note': 'some_note',
    'model': 'BGCN',
    'dataset_name': 'NetEase',
    'task': 'tune',
    'eval_task': 'test',

    ## search hyperparameters
    #  'lrs': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    #  'message_dropouts': [0, 0.1, 0.3, 0.5],
    #  'node_dropouts': [0, 0.1, 0.3, 0.5],
    #  'decays': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],

    ## optimal hyperparameters 
    'lrs': [3e-4],
    'message_dropouts': [0],
    'node_dropouts': [0], 
    'decays': [1e-7],

    ## hard negative sample and further train
    'sample': 'simple',
    # 'sample': 'hard',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.4, 0.4], # probability 0.8
    # 'conti_train': '/data/prj2020/zss/gpu-bgcn/log/NetEase/BGCN_tune/09-03-08-03-56-some_note/1_a79c1d_Recall@20.pth',
    # 'conti_train': '/data/prj2020/zss/gpu-bgcn/log/NetEase/BGCN_tune/12-27-11-45-21-some_note/1_f9c647_Recall@20.pth',
    # 'conti_train': '/data/prj2020/zss/gpu-bgcn/log/steamgame/BGCN_tune/12-28-16-01-56-some_note/1_fee9a1_Recall@20.pth',

    ## other settings
    'epochs': 5,
    'early': 20,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    ## test path
    # 'test':['/data/prj2020/zss/gpu-bgcn/log/NetEase/BGCN_tune/12-27-15-23-06-some_note']
    # 'test': ['/data/prj2020/zss/gpu-bgcn/log/steamgame/BGCN_tune/12-29-23-02-21-some_note/']
    # 'test': ['/data/prj2020/zss/gpu-bgcn/log/steamgame/BGCN_tune/12-28-20-57-43-some_note/']

}


