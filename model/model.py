import torch.nn as nn
import transformers
from transformers import BertConfig

class BertModel(nn.Module):
    def __init__(self, ptm_path="bert-base-chinese/", hidden_size=768, class_num=3, dropout=0.2):
        super(StudentBertModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(ptm_path)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all=None):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                output_attentions=True,output_hidden_states=True)
        last_hidden_state=bert_output['last_hidden_state']
        all_hidden_states=bert_output['hidden_states']
        all_attentions=bert_output['attentions']
        pooler_output = bert_output['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        if(output_all==True):
            return logits, all_hidden_states, all_attentions
        return logits


class StudentBertModel(nn.Module):
    def __init__(self, config_path="bert-distill/config.json", hidden_size=768, class_num=3, dropout=0.2):
        super(StudentBertModel, self).__init__()
        config=BertConfig.from_pretrained(config_path)
        self.bert = transformers.BertModel(config)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all=None):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                output_attentions=True, output_hidden_states=True)
        last_hidden_state = bert_output['last_hidden_state']
        hidden_states = bert_output['hidden_states']
        attentions = bert_output['attentions']
        pooler_output = bert_output['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        if (output_all == True):
            return logits, hidden_states, attentions
        return logits

