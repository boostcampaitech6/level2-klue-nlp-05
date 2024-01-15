import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast
import pandas as pd
from tqdm import tqdm


class Custom_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model.model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.model.last_dense_layer_dropout_prob),
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs

class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = ['[PER]', '[ORG]', '[DAT]', '[LOC]', '[POH]', '[NOH]']
        self.tokenizer.add_tokens(self.new_tokens)
        self.LABEL_TO_ID = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, \
                'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, \
                'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, \
                'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, \
                'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
        
    def token_location(self, list1, list2):
        for idx in range(len(list1) - len(list2) + 1):
            if list1[idx:idx + len(list2)] == list2:
                index = idx
                return idx
            
    def add_marker_tokens(self, sentence, subj_type, obj_type, ss, se, os, oe):
        subj_type , obj_type = f"[{subj_type}]", f"[{obj_type}]"
        new_sentence=''
        if ss < os:
            new_sentence += sentence[ :ss]
            new_sentence += f"@*{subj_type}*{sentence[ss:se+1]}@"
            new_sentence += sentence[se+1:os]
            new_sentence += f"#^{obj_type}^{sentence[os:oe+1]}#"
            new_sentence += sentence[oe+1:]
        else:
            new_sentence += sentence[ :os]
            new_sentence += f"#^{obj_type}^{sentence[os:oe+1]}#"
            new_sentence += sentence[oe+1:ss]
            new_sentence += f"@*{subj_type}*{sentence[ss:se+1]}@"
            new_sentence += sentence[se+1:]

        sents = self.tokenizer.tokenize(new_sentence)
        new_ss= self.token_location(sents,['@',"*"])
        new_os= self.token_location(sents,['#',"^"])

        return new_sentence, new_ss +1 , new_os +1
    
    def read(self, train_dataset):

        new_sentence_list=[]
        new_ss_list=[]
        new_os_list=[]
        for _, d in tqdm(train_dataset.iterrows()):
            ss, se = int(d['subject_start_idx']), int(d['subject_end_idx'])
            os, oe = int(d['object_start_idx']), int(d['object_end_idx'])

            new_sentence, new_ss, new_os= self.add_marker_tokens(d['sentence'], d['subject_type'], d['object_type'], ss, se, os, oe)
            new_sentence_list.append(new_sentence)
            new_ss_list.append(new_ss)
            new_os_list.append(new_os)
            
        tokenized_sentences = self.tokenizer(
        new_sentence_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False
        )
        
        
        tokenized_sentences['ss'] = torch.tensor(new_ss_list)
        tokenized_sentences['os'] = torch.tensor(new_os_list)

        return tokenized_sentences
    
