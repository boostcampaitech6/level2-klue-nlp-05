import pandas as pd
import numpy as np
import random
import torch
import argparse
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
import wandb

from model import CustomModel
from load_data import RE_Dataset, load_data, tokenized_dataset, label_to_num
from metrics import compute_metrics


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")

  set_seed(conf.utils.seed)

  wandb.login()
  wandb.init(project=conf.wandb.project_name)

  # load model and tokenizer
  MODEL_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # add special tokens
  special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  # load dataset
  train_dataset = load_data(conf.path.train_path)
  validation_dataset = load_data(conf.path.validation_path)

  train_label = label_to_num(train_dataset['label'].values)
  validation_label = label_to_num(validation_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_validation = tokenized_dataset(validation_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_validation, validation_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model = CustomModel(conf, config=model_config)
  # resize token embeddings
  model.encoder.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

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
    metric_for_best_model='micro f1 score',
    report_to='wandb',
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
  save_path = f"./best_model/model.pt"
  torch.save(model.state_dict(), save_path)
  
  wandb.finish()