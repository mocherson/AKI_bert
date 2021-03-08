import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from pytorch_transformers import BertPreTrainedModel, BertModel

class BertForDocClassification(BertPreTrainedModel):
    def __init__(self, config, pooling='max' ,dropout_prob=0):
        super(BertForDocClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pooling = pooling

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bn = nn.BatchNorm1d(768)

        self.init_weights()

    def forward(self, input_ids, max_seqs_per_doc=200, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        dev = next(self.parameters()).device
        n_sentences=[len(x) for x in input_ids]
        print(n_sentences)
        if self.training:
            for i, x in enumerate(input_ids):
                if len(x)>max_seqs_per_doc:
                    n_sentences[i] = max_seqs_per_doc 
                    ind = np.random.choice(len(x), size=max_seqs_per_doc, replace=False)
                    input_ids[i] = input_ids[i][ind]
                    if token_type_ids is not None:
                        token_type_ids[i] = token_type_ids[i][ind]
                    if attention_mask is not None:
                        attention_mask[i] = attention_mask[i][ind]

        input_ids = torch.from_numpy(np.concatenate(input_ids, axis=0)).to(dev)
        token_type_ids = torch.from_numpy(np.concatenate(token_type_ids, axis=0)).to(dev) if token_type_ids is not None else None
        attention_mask = torch.from_numpy(np.concatenate(attention_mask, axis=0)).to(dev) if attention_mask is not None else None
        
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[0][:,0,:]
#         pooled_output = self.bn(pooled_output)

        pooled_output = self.dropout(pooled_output)
        if self.pooling=='max':
            pooled_output = torch.stack([t.max(0)[0] for t in pooled_output.split(n_sentences)])
        elif self.pooling=='avg':
            pooled_output = torch.stack([t.mean(0) for t in pooled_output.split(n_sentences)])
        
        logits = self.classifier(pooled_output)
        prob = F.softmax(logits,dim=1)
        
#         prob = torch.stack([t.mean(0) for t in logits.split(n_sentences)])
#         prob = torch.cat([1-prob,prob], dim=1)

        outputs = (logit, prob, pooled_output) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
        