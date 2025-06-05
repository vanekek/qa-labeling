from transformers import BertModel
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F


## Stolen from transformer code base without any noble intention.
class CustomBert(nn.Module):

    def __init__(self, config):
        super(CustomBert, self).__init__()
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear  = nn.Linear(config.hidden_size,1024)
        self.classifier = nn.Linear(1024, config.num_labels)

        # self.init_weights() - adding is needed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        lin_output = F.relu(self.linear(pooled_output))
                                               
        lin_output = self.dropout(lin_output)
        logits = self.classifier(lin_output)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs