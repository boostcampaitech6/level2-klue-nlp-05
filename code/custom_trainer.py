from transformers import Trainer
from loss import SmoothFocalCrossEntropyLoss

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")

    loss_func = SmoothFocalCrossEntropyLoss(smoothing=0.1, gamma=2.0)
    loss = loss_func(logits, labels)

    return (loss, outputs) if return_outputs else loss