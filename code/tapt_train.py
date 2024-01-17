from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from dataset_utils import load_pretrain_data, tokenized_dataset, tokenized_dataset_pretrain
from omegaconf import OmegaConf
from datasets import RE_Dataset_pretrain
from train import set_seed

import argparse
import torch
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="tapt_config")

    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    # wandb log출력 x
    os.environ["WANDB_DISABLED"] = "true"

    set_seed(conf.utils.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model.model_name)

    # load dataset (train data + test data)
    pretrain_dataset = load_pretrain_data(conf.path.train_path, conf.path.test_path)

    print("pretrain_data len : ", len(pretrain_dataset))

    tokenized_pretrain = tokenized_dataset_pretrain(pretrain_dataset, tokenizer)

    print("tokenized data len : ", len(tokenized_pretrain))

    RE_pretrain_dataset = RE_Dataset_pretrain(tokenized_pretrain, None)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 15% token masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # load config
    model_config = AutoConfig.from_pretrained(conf.model.model_name)

    model = AutoModelForMaskedLM.from_pretrained(conf.model.model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./tapt/klue_roberta_large',
        save_total_limit=conf.utils.top_k,
        save_steps=conf.train.save_steps,
        save_strategy="steps",
        num_train_epochs=conf.train.epochs,
        learning_rate=conf.train.learning_rate,
        per_device_train_batch_size=conf.train.batch_size,
        per_device_eval_batch_size=conf.train.batch_size,
        logging_dir='./logs',
        logging_steps=conf.train.logging_steps,
        resume_from_checkpoint=True,
        fp16=conf.train.fp16,
        gradient_accumulation_steps=conf.train.gradient_accumulation_step
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_pretrain_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("./tapt/klue_roberta_large")