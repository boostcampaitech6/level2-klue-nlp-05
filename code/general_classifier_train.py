from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from omegaconf import OmegaConf
from dataset_utils import load_data, label_to_num, tokenized_dataset
from custom_datasets import RE_Dataset
from metrics import compute_metrics
from model import CustomModel

import numpy as np
import pandas as pd
import argparse
import random
import torch
import wandb
import os


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def no_relation_delete(train_path, dev_path):
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    
    train_grouped = train_df.groupby('label')
    dev_grouped = dev_df.groupby('label')

    new_train_df = pd.DataFrame()
    new_dev_df = pd.DataFrame()

    for name, group in train_grouped:
       if name != 'no_relation':
          new_train_df = pd.concat([new_train_df, group], axis=0)
          
    for name, group in dev_grouped:
       if name != 'no_relation':
          new_dev_df = pd.concat([new_dev_df, group], axis=0)
      
    new_train_df = new_train_df.sort_values(by='id', ascending=True)
    new_dev_df = new_dev_df.sort_values(by='id', ascending=True)
    
    new_train_df.to_csv('../dataset/train/general_train_true_labels.csv')
    new_dev_df.to_csv('../dataset/train/general_dev_true_labels.csv')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")

  set_seed(conf.utils.seed)

  # 'no_relation' label 삭제
  no_relation_delete(conf.path.train_path, conf.path.dev_path)

  wandb.login()
  wandb.init(project=conf.wandb.project_name,
             entity='level2-klue-nlp-05',
             name=f'{conf.wandb.curr_ver} (General)')

  # load model and tokenizer
  MODEL_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # 스페셜 토큰 추가
  special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  # load dataset(for True labels)
  # train_dataset = load_data('../dataset/train/general_train_true_labels.csv')
  # dev_dataset = load_data('../dataset/train/general_dev_true_labels.csv')

  # load dataset(for ALL labels)
  train_dataset = load_data('../dataset/train/train_final.csv')
  dev_dataset = load_data('../dataset/train/dev_final.csv')

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)

  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model = CustomModel(conf, config=model_config)
  
  # 스페셜 토큰 추가로 인한 모델의 임베딩 크기 조정
  model.encoder.resize_token_embeddings(len(tokenizer))  
  
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
    fp16=conf.train.fp16
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
  model.save_pretrained('./best_model/General')
  
  wandb.finish()