# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:25:57 2020

@author: Jinliang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class FinetuningConfig():
    """Fine-tuning hyperparameters."""

    def __init__(self, do_train=True, do_predict=True, training_config=None, test_config=None,
                 train_epochs=1.0, top_n=3, model_dir=None, vocab_file=None, bert_config_file=None,
                 output_dir='output'):

        # basic parameters
        self.do_train = do_train
        self.do_predict = do_predict
        self.output_dir = output_dir
        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file
        self.init_checkpoint = model_dir
        self.test_config = test_config
        self.training_config = training_config
        # other parameters
        self.do_lower_case = True
        self.max_seq_length = 384

        self.train_batch_size = 12
        self.predict_batch_size = 8
        self.learning_rate = 3e-5
        self.num_train_epochs = train_epochs

        self.warmup_proportion = 0.1
        self.iterations_per_loop = 4000
        self.save_checkpoints_steps = 4000
        """
        ----Parameter: save_checkpoints_steps---- 
            It determines how many models you want to save during training

            Suppose you have (1) 6400 samples training data (2) 64 batch_size (3) 3 training_epoch
            Then for every epoch: you will get: 6400/64 = 100 batches to feed the model
            For whole training process: you will get: 100*3 = 300 steps (one batch means one step)

            If save_checkpoints_steps = 100, then you will get 4 saved models after training:
            model-step-0, model-step-100, model-step-200, model-step-300

            If save_checkpoints_steps is small:
                (1) Models will be saved more frequently during training.
                (2) But it will cost more disk space because size of BERT is big.
            If save_checkpoints_steps is big enough:
                (1) Models will be saved only twice.
                (2) But if your computer goes wrong during training, no intermediate models will be saved.
        """
        # QA parameter
        self.doc_stride = 128  # the size of sliding window, splitting up a long document into chunks
        self.max_query_length = 64  # Questions longer than this will be truncated to this length. No padding
        self.max_answer_length = 64  # the start and end predictions are not conditioned on one another.
        self.n_best_size = top_n  # The total number of n-best predictions to generate
        self.version_2_with_negative = False  # If true, the SQuAD examples contain some that do not have an answer.
        self.null_score_diff_threshold = 0.0  # If null_score-best_non_null is greater than the threshold predict null.

        self.verbose_logging = False  # If true, all of the warnings related to data processing will be printed.
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.master = None
        self.num_tpu_cores = 8
