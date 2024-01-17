from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from omegaconf import OmegaConf
from dataset_utils import load_data, label_to_num, tokenized_dataset, tokenized_dataset_xlm
from datasets import RE_Dataset
from metrics import compute_metrics
import numpy as np
import pandas as pd
import argparse
import random
import torch
import wandb
import os

from config.config import call_config
import torch.nn as nn
import torch.nn.functional as F
from custom_model import CustomModel, CustomModel2
from custom_tokenizer import Processor

def set_seed(seed:int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
    

if __name__ == '__main__':
  conf = call_config()

  set_seed(42)

  wandb.login()
  wandb.init(project=conf.wandb.project_name,
             entity='level2-klue-nlp-05',
             name=conf.wandb.curr_ver)

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
  if conf.custom_model==1:
    model = CustomModel(MODEL_NAME, config=model_config)
  elif conf.custom_model==2:
    model = CustomModel2(MODEL_NAME, config=model_config)
  else:
    model_config.num_labels = 30
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
   
  model.parameters
  model.to(device)

  
  def model_init():
    return model
  
  def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]),
        "seed": trial.suggest_int("seed", 40, 42)
    }
  
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
    fp16=conf.train.fp16,
    gradient_accumulation_steps=2
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_dev_dataset,
    compute_metrics=compute_metrics
  )
  # best_trials = trainer.hyperparameter_search(n_trials=20, direction="maximize",backend="optuna", hp_space=optuna_hp_space) 
  # train model
  trainer.train()
  
  if conf.model.add_entity_token:
    save_path = f"./model_save/{conf.model.model_name.replace('/', '_')}_Max-epoch:{conf.train.epochs}_Batch-size:{conf.train.batch_size}"
    torch.save(model.state_dict(), save_path)
  else:
    model.save_pretrained('./best_model')
  wandb.finish()
