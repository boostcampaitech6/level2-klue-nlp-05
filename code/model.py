import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

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
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, ss=None, os=None):
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