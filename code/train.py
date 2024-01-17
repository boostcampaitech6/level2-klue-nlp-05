from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from omegaconf import OmegaConf
from dataset_utils import load_data, label_to_num, tokenized_dataset, tokenized_dataset_xlm
from datasets import RE_Dataset
from metrics import compute_metrics

import numpy as np
import argparse
import random
import torch
import wandb
import os

from config.config import call_config
import torch.nn as nn
import torch.nn.functional as F
from custom_model import CustomModel
from custom_tokenizer import Processor

def set_seed(seed:int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
    
class SmoothFocalCrossEntropyLoss(nn.Module):
  def __init__(self, smoothing=0.1, gamma=2.0):
    super(SmoothFocalCrossEntropyLoss, self).__init__()
    self.smoothing = smoothing
    self.gamma = gamma

  def forward(self, input, target):
    log_prob = F.log_softmax(input, dim=-1)
    prob = torch.exp(log_prob)
    weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
    
    # focal loss weight
    focal_weight = (1 - prob).pow(self.gamma)
    weight *= focal_weight

    loss = (-weight * log_prob).sum(dim=-1).mean()
    return loss

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")

    loss_func = SmoothFocalCrossEntropyLoss(smoothing=0.1, gamma=2.0)
    loss = loss_func(logits, labels)

    return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
  conf = call_config()

  set_seed(42)

  wandb.login()
  wandb.init(project=conf.wandb.project_name, name=conf.wandb.version_name)

  # load model and tokenizer
  MODEL_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("../dataset/train/train.csv", train=True)
  dev_dataset = load_data("../dataset/train/dev.csv")

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  if MODEL_NAME.split('-')[0] == 'xlm':
    tokenized_train = tokenized_dataset_xlm(train_dataset, tokenizer)
  else:
    if conf.custom_model:
      tokenized_train = Processor(conf, tokenizer).read(train_dataset, train=True)
      tokenized_dev = Processor(conf, tokenizer).read(dev_dataset)
    else:
      tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  
  # Split train dataset into train, valid.
  # RE_train_dataset, RE_dev_dataset = torch.utils.data.random_split(RE_train_dataset, [int(len(RE_train_dataset)*0.8), len(RE_train_dataset)-int(len(RE_train_dataset)*0.8)])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  if conf.custom_model:
    model = CustomModel(MODEL_NAME, config=model_config)
  else:
    model_config.num_labels = 30
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
  # print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=conf.utils.top_k,  # number of total save model.
    save_steps=conf.train.save_steps,                 # model saving step.
    num_train_epochs=conf.train.epochs,              # total number of training epochs
    learning_rate=conf.train.learning_rate,               # learning_rate
    per_device_train_batch_size=conf.train.batch_size,  # batch size per device during training
    per_device_eval_batch_size=conf.train.batch_size,   # batch size for evaluation
    warmup_steps=conf.train.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=conf.train.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=conf.train.logging_steps,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps=conf.train.eval_steps,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model="micro f1 score",
    report_to="wandb",
    fp16=True
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_dev_dataset,
    compute_metrics=compute_metrics
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
  
  wandb.finish()
