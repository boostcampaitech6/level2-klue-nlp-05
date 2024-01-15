import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast

class CustomModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model.model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, entity_token_ids=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0]  # Shape: [batch_size, sequence_length, hidden_size]
        
        ss, os = [], []
        for input_id, (id1, id2) in zip(input_ids, entity_token_ids):
            idx1, idx2 = (input_id == id1).nonzero(as_tuple=True)[0], (input_id == id2).nonzero(as_tuple=True)[0]
            ss.append(idx1)
            os.append(idx2)

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb, os_emb = pooled_output[idx, ss], pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs

        return outputs


    # @autocast()
    # def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
    #     outputs = self.encoder(
    #         input_ids,
    #         attention_mask=attention_mask,
    #     )
    #     pooled_output = outputs[0]
    #     idx = torch.arange(input_ids.size(0)).to(input_ids.device)
    #     ss_emb = pooled_output[idx, ss]
    #     os_emb = pooled_output[idx, os]
    #     h = torch.cat((ss_emb, os_emb), dim=-1)
    #     logits = self.classifier(h)
    #     outputs = (logits,)
    #     if labels is not None:
    #         loss = self.loss_fnt(logits.float(), labels)
    #         outputs = (loss,) + outputs
    #     return outputs