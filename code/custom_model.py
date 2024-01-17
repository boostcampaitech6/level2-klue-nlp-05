import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast
from transformers import PreTrainedModel

from torch.nn import functional as F

class CrossEntropywithLabelSmoothing(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        assert preds.size(0) == target.size(0)
        K = preds.size(-1)  # 전체 클래스의 갯수
        log_probs = F.log_softmax(preds, dim=-1)
        avg_log_probs = (-log_probs).sum(-1).mean()
        ce_loss = F.nll_loss(log_probs, target)
        ce_loss_w_soft_label = (1 - self.epsilon) * ce_loss + self.epsilon / K * avg_log_probs
        return ce_loss_w_soft_label
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        
        return F_loss.mean()


class CustomModel(PreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config=config)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = FocalLoss()
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 30)
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_output = outputs[0]
        pooled_output = outputs[1]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = last_output[idx, ss]
        os_emb = last_output[idx, os]
        h = torch.cat((pooled_output,ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs