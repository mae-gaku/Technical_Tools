#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.input_size = (640, 640)  # (height, width)
        self.data_dir = "data/custom"
        self.train_ann = "custom_train.json"
        self.val_ann = "custom_val.json"

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0        
        
        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 50
        
        self.act = "silu"
