import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class CustomModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model.model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = FocalLoss()
        self.fc_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids, attention_mask=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        cls_emb = pooled_output[:, 0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        fc_output = self.fc_layer(torch.cat((ss_emb, os_emb), dim=-1))
        h = self.fc_layer(torch.cat((fc_output, cls_emb), dim=-1))
        logits = self.classifier(h)
        outputs = (logits,)
        
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs

        return outputs
    
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.query_layer = nn.Linear(hidden_size, hidden_size)
#         self.key_layer = nn.Linear(hidden_size, hidden_size)
#         self.value_layer = nn.Linear(hidden_size, hidden_size)

#     def forward(self, query, key, value):
#         query = self.query_layer(query)
#         key = self.key_layer(key)
#         value = self.value_layer(value)

#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
#         attention_weights = torch.softmax(attention_scores, dim=-1)
#         attended_value = torch.matmul(attention_weights, value)

#         return attended_value

# class CustomModel(nn.Module):
#     def __init__(self, args, config):
#         super().__init__()
#         self.args = args
#         self.encoder = AutoModel.from_pretrained(args.model.model_name, config=config)
#         hidden_size = config.hidden_size
#         self.loss_fnt = FocalLoss()
#         self.attention = Attention(hidden_size)
#         self.fc_layer = nn.Sequential(
#             nn.Linear(2 * hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(p=0.1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, 30)
#         )

#     @autocast()
#     def forward(self, input_ids, attention_mask=None, labels=None, ss=None, os=None):
#         outputs = self.encoder(
#             input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = outputs[0]
#         cls_emb = pooled_output[:, 0]
#         idx = torch.arange(input_ids.size(0)).to(input_ids.device)
#         ss_emb = pooled_output[idx, ss]
#         os_emb = pooled_output[idx, os]
#         fc_output = self.fc_layer(torch.cat((ss_emb, os_emb), dim=-1))
#         attention_output = self.attention(cls_emb.unsqueeze(1), fc_output.unsqueeze(1), fc_output.unsqueeze(1))
#         h = attention_output.squeeze(1)
#         logits = self.classifier(h)
#         outputs = (logits,)

#         if labels is not None:
#             loss = self.loss_fnt(logits.float(), labels)
#             outputs = (loss,) + outputs

#         return outputs