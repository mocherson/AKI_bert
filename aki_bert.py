from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule, BertModel, BertForSequenceClassification

from readdata import AKIdata
from model import BertForDocClassification
from utils import save_obj, load_obj, StratifiedBatchSampler, BalancedBatchSampler

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  
fhlr = logging.FileHandler('aki_bert.log', 'w')
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)


def train_val_test(args, model, train_loader, val_loader, test_loader, missing_keys=()  ):
    """ Train the model """

    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if n in missing_keys], 'lr': 0.001},
#         {'params': [p for n, p in model.named_parameters() if n not in missing_keys]}
#         ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
#     wt = torch.Tensor([2785/16560, 13775/16560])   # the classification task is class-imbanlanced
#     wt = torch.Tensor([0.25, 0.75])
    loss_func = nn.CrossEntropyLoss(weight=wt).to(args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_loader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.batch_size)
    logger.info("  Total train batch size ( distributed & accumulation) = %d",
                   args.batch_size * args.gradient_accumulation_steps )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    res_eval=[]   
    res_test=[] 
    eval_best=0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            targets = torch.LongTensor(batch['label']).to(args.device)
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],token_type_ids=batch['segment_ids'], max_seqs_per_doc=args.max_seq_per_doc )
            prob = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = loss_func(prob, targets)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
              
            if (step+1)%500==0:
                logger.info('step:%d,  training loss: %s', global_step, tr_loss / global_step)
                results_eval = evaluate(args, model, val_loader)
                results_test = evaluate(args, model, test_loader)
                res_eval.append(results_eval)
                res_test.append(results_test)
                
                if results_eval[-4]>eval_best:
                    output_dir = os.path.join(pretrained_path, 'AKI/model_{}_ep{}_{}pool_wd0.001_drop{}{}'.format(args.sampling, args.num_train_epochs, args.pooling, args.dropout,'_oneseq' if args.as_one_sequence else ''))
                    if not os.path.exists(os.path.join(pretrained_path,'AKI')):
                        os.makedirs(os.path.join(pretrained_path,'AKI'))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

#                 output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(args.bert_model.split('/')[-2] if args.bert_model!='bert-base-uncased' else 'bert-base-uncased',global_step))
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#                 model_to_save.save_pretrained(output_dir)
#                 torch.save(args, os.path.join(output_dir, 'training_args.bin'))
#                 logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step, np.array(res_eval), np.array(res_test)
                
def evaluate(args, model, val_dataloader):
    model.eval()

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(val_dataloader.dataset))
    logger.info("  Batch size = %d", args.batch_size)

    loss_func = nn.CrossEntropyLoss(weight=wt).to(args.device)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):                
            targets = torch.LongTensor(batch['label']).to(args.device)
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],token_type_ids=batch['segment_ids'] )
            logits = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = loss_func(logits, targets)

            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            preds.append(logits)
            labels += batch['label']

        prob = F.softmax(torch.cat(preds, dim=0), dim=1).cpu().numpy()
        eval_loss = eval_loss / nb_eval_steps
        score = prob[:,1]
        preds = prob.argmax(axis=1)
    
    acc, precision, recall, f1, auc = accuracy_score(labels,preds), precision_score(labels,preds), recall_score(labels,preds),  \
                           f1_score(labels,preds), roc_auc_score(labels, score)
    
    logger.info("***** Eval results  *****")
    logger.info("  loss = %s", eval_loss)
    logger.info("  accuracy = %s", acc)
    logger.info("  precision = %s", precision)
    logger.info("  recall = %s", recall)
    logger.info("  f1 = %s", f1)
    logger.info("  auc = %s", auc)

    return eval_loss, acc, precision, recall, f1, auc

def get_bert_output(args, model, dataloader):
    model.eval()
    out=[]
    labels=[]
    icustay_id=[]
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['input_mask'],token_type_ids=batch['segment_ids'] )
            out.append(outputs)
            labels += batch['label']
            icustay_id += batch['icustay_id']
        out = [torch.cat(x,dim=0).cpu().numpy() for x in zip(*out)]
    return out, labels, icustay_id


parser = argparse.ArgumentParser(description='Bert for AKI notes')

## Required parameters
parser.add_argument("--data_dir",default='./data/',type=str,
              help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default='./AKI_bert_simple_lm_notes/', type=str,
              help="Bert pre-trained model")
parser.add_argument("--output_dir",default='./checkpoint/',type=str,
              help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--max_seq_length",default=32,type=int,
              help="The maximum total input sequence length after WordPiece tokenization. \n"
                 "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--max_seq_per_doc",default=200,type=int,
              help="The maximum number of sequences per documendation, \n"
                 "docs with more sentences are randomly sampled to max_seq_per_doc sentences .")
parser.add_argument("--do_train",action='store_true',help="Whether to run training.")
parser.add_argument("--save_bert_output",action='store_true',help="Whether to save the output of bert")
parser.add_argument("--as_one_sequence",action='store_true',help="Whether to see a note as a sequence")
parser.add_argument("--batch_size",default=4,type=int,help="Total batch size for training.")
parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",default=3.0,type=float,help="Total number of training epochs to perform.")
parser.add_argument("--dropout",default=0,type=float,help="drop out rate")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--gpu",type=int,default=-1,help="gpu for training")
parser.add_argument('--seed',type=int,default=42,help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
              help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--sampling",default='wt',type=str,
              help="sampling method, wt, SBS, US, DS")
parser.add_argument("--pooling",default='max',type=str,
              help="pooling method, max,avg")

args = parser.parse_args()

lr = args.learning_rate
seed = args.seed 
batch_size=args.batch_size 
savepath = args.output_dir
path = args.data_dir
pretrained_path = args.bert_model
max_seq_length = args.max_seq_length
epochs = args.num_train_epochs
if args.gpu<0:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
else:
    torch.cuda.set_device(args.gpu)
    args.device = torch.device("cuda", args.gpu)    

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = AKIdata(path = path, max_seq_length=max_seq_length, pretrained_path=pretrained_path, as_one_sequence=args.as_one_sequence)
tr_set, val_set, te_set, tr_val_set = dataset.split()

if args.sampling=='wt':
    wt = torch.Tensor([len(dataset)/dataset.label_count[0], len(dataset)/dataset.label_count[1]])
    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=4)
elif args.sampling=='SBS':
    batch_sampler = StratifiedBatchSampler(tr_set.data_str.iloc[tr_set.index]['aki_label'], batch_size)
    wt = torch.Tensor([batch_size/batch_sampler.label_count_per_batch[0], batch_size/batch_sampler.label_count_per_batch[1]])
    train_loader = DataLoader(tr_set,  batch_sampler=batch_sampler, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=0)
elif args.sampling=='US':
    batch_sampler = BalancedBatchSampler(tr_set.data_str.iloc[tr_set.index]['aki_label'], batch_size,downsample=False)
    wt=None
    train_loader = DataLoader(tr_set, batch_sampler=batch_sampler, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=0)
elif args.sampling=='DS':
    wt=None
    batch_sampler = BalancedBatchSampler(tr_set.data_str.iloc[tr_set.index]['aki_label'], batch_size,downsample=True)
    train_loader = DataLoader(tr_set,  batch_sampler=batch_sampler, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=0)
    
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=4)
test_loader = DataLoader(te_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {key: [d[key] for d in x] for key in x[0]}, pin_memory=True, num_workers=4)

label_list = dataset.label_list
num_labels = len(label_list)


if os.path.isfile(pretrained_path+WEIGHTS_NAME) or pretrained_path in ('bert-base-uncased',):
    model,loading_info = BertForDocClassification.from_pretrained(pretrained_path,num_labels=num_labels,dropout_prob=args.dropout, \
                                          pooling=args.pooling, from_tf= False, output_loading_info=True)
    missed = loading_info['missing_keys']
else:
    model = BertForDocClassification.from_pretrained(pretrained_path,num_labels=num_labels,dropout_prob=args.dropout,\
                                                     pooling=args.pooling, from_tf= True) 
    missed = ()
model.to(args.device)
    
if args.do_train:
    global_step, tr_loss, res_val, res_test = train_val_test(args, model, train_loader, val_loader, test_loader, missed )
    res_df = pd.DataFrame(np.concatenate([res_val,res_test], axis=1),columns=['loss_val','acc_val', 'precision_val', 'recall_val', 'f1_val',
                             'AUC_val', 'loss_test', 'acc_test','precision_test','recall_test','f1_test','AUC_test']) 
    if not os.path.exists(os.path.join(pretrained_path,'AKI')):
        os.makedirs(os.path.join(pretrained_path,'AKI'))
    res_df.to_csv(os.path.join(pretrained_path,'AKI/AKI_res_{}_ep{}_{}pool_wd0.001_drop{}{}.csv'.format(args.sampling, args.num_train_epochs, args.pooling, args.dropout,'_oneseq' if args.as_one_sequence else '')))
if args.save_bert_output:   
#     model = BertForDocClassification.from_pretrained(pretrained_path,num_labels= num_labels, 
#                                      from_tf= False )     
#     model.to(args.device)
    tr_out, tr_label, tr_id = get_bert_output(args, model, train_loader)
    val_out, val_label, val_id = get_bert_output(args, model, val_loader)
    test_out, test_label, test_id = get_bert_output(args, model, test_loader)  
    if not os.path.exists(os.path.join(pretrained_path,'AKI')):
        os.makedirs(os.path.join(pretrained_path,'AKI'))
    save_obj((tr_out, tr_label, tr_id), os.path.join(pretrained_path,'AKI/poolfea_from_bert_train_MSL%d%s.pkl'%(max_seq_length, '_oneseq' if args.as_one_sequence else '')))
    save_obj((val_out, val_label, val_id),os.path.join(pretrained_path,'AKI/poolfea_from_bert_val_MSL%d%s.pkl'%(max_seq_length, '_oneseq' if args.as_one_sequence else '')))
    save_obj((test_out, test_label, test_id),os.path.join(pretrained_path,'AKI/poolfea_from_bert_test_MSL%d%s.pkl'%(max_seq_length, '_oneseq' if args.as_one_sequence else '')))            


    
             
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            