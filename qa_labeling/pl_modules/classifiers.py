import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from ..utils import TARGETS_NUM


class CustomBert(nn.Module):
    def __init__(self, config):
        super(CustomBert, self).__init__()
        self.num_labels = TARGETS_NUM

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert_config = BertConfig.from_pretrained("bert-base-uncased")

        self.dropout = nn.Dropout(config["model"]["hidden_dropout_prob"])
        self.linear = nn.Linear(
            self.bert_config["hidden_size"], config["model"]["hidden_size"]
        )
        self.classifier = nn.Linear(config["model"]["hidden_size"], self.num_labels)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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

        logits = self.classifier(lin_output)

        return logits
