from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from label_num import label_to_num
from load_data import *
from metrics import *

import pickle as pickle
import numpy as np
import random
import torch
import wandb


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train():
  # 실행할때마다 버전 바꿔주기
  wandb.init(project="level2_RE", entity="version==X.X.X")

  set_seed(42)
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("../dataset/train/train.csv")
  dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  if MODEL_NAME.split('-')[0] == 'xlm':
    tokenized_train = tokenized_dataset_xlm(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset_xlm(dev_dataset, tokenizer)
  else:
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

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, 
                                                              config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,

    report_to="wandb" #wandb log 찍어
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function    
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')

  wandb.finish()

def main():
  train()

if __name__ == '__main__':
  main() 