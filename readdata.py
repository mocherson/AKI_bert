from torch.utils.data import Dataset, DataLoader
import pickle as pk
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from pytorch_transformers import BertTokenizer
import copy
from utils import InputExample, convert_example_to_feature
        
        
class AKIdata(Dataset):    
    def __init__(self, path = './data/', max_seq_length=256, pretrained_path= '../pretrained_bert_tf/bert_pretrain_output_all_notes_150000/',
                as_one_sequence=True):
        self.path=path
        notes = pd.read_csv(path+'formatted_notes.csv',index_col=0)  
        events = pd.read_csv(path+'icunoteevents.csv',usecols = ['note_row','icustay_id'])        
        notes = pd.merge(notes, events, left_index=True, right_on='note_row')
        self.notes = notes.loc[notes['text'].str.len()>0]
        combined = pd.read_csv(path+'combined_data.csv')
        data_str = combined.drop(['Unnamed: 0', 'Unnamed: 0.1', 'text','cui','creatinine_max_day_3'], axis=1)
        self.data_str = pd.get_dummies(data_str,columns=['gender','ethnicitynewfactor'])
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.index = range(len(self.data_str))
        self.as_one_sequence = as_one_sequence
        self.label_list = self.data_str['aki_label'].unique()
        
#         if as_one_sequence:
#             input_ids, input_mask, segment_ids = [], [], []
#             for i in self.index:
#                 example = InputExample(guid=i, text_a=data.iloc[i]['text'], text_b=None, label=data.iloc[i]['aki_label'])
#                 in_ids, in_mask, seg_ids = convert_example_to_feature(example, self.max_seq_length, self.tokenizer)
#                 input_ids.append( in_ids)
#                 input_mask.append(in_mask)
#                 segment_ids.append( seg_ids )
#             self.input_ids = np.array(input_ids)
#             self.input_mask= np.array(input_mask)
#             self.segment_ids= np.array(segment_ids)
#         else:
#             self.input_ids, self.input_mask, self.segment_ids = [], [], []
#             for i in self.index:
#                 sentences = data.iloc[i]['text'].strip().split('\n')
#                 ms_ids, ms_mask, msseg_ids = [], [], []
#                 for s in sentences:
#                     if len(s)>0:
#                         example = InputExample(guid=i, text_a=s, text_b=None, label=None)
#                         s_ids, s_mask, sseg_ids = convert_example_to_feature(example, self.max_seq_length, self.tokenizer)
#                         ms_ids.append(s_ids)
#                         ms_mask.append(s_mask)
#                         msseg_ids.append(sseg_ids)
#                 self.input_ids.append(np.array(ms_ids))
#                 self.input_mask.append(np.array(ms_mask))
#                 self.segment_ids.append(np.array(msseg_ids))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self,idx):
        rec = self.data_str.iloc[self.index[idx]]
        icustay_id = rec['icustay_id'] 
        stru = np.array(rec.drop(['aki_label','icustay_id','subject_id','hadm_id']),dtype=float)   
        note = self.notes.query('icustay_id==@icustay_id')['text'].drop_duplicates()
        
        if self.as_one_sequence:
            input_ids, input_mask, segment_ids = [], [], []
            for nt in note:
                example = InputExample(guid=0, text_a=nt, text_b=None, label=None)
                in_ids, in_mask, seg_ids = convert_example_to_feature(example, self.max_seq_length, self.tokenizer)
                input_ids.append( in_ids)
                input_mask.append(in_mask)
                segment_ids.append( seg_ids )                             
        else:
            input_ids, input_mask, segment_ids = [], [], []
            for nt in note:
                sentences = nt.strip().split('\n')
                for s in sentences:
                    if len(s)>0:
                        example = InputExample(guid=0, text_a=s, text_b=None, label=None)
                        in_ids, in_mask, seg_ids = convert_example_to_feature(example, self.max_seq_length, self.tokenizer)
                        input_ids.append( in_ids)
                        input_mask.append(in_mask)
                        segment_ids.append( seg_ids ) 
                        
        sample={'label':rec['aki_label'], 'data_str': stru, 'input_ids':np.array(input_ids), 'input_mask': np.array(input_mask), 
              'segment_ids':np.array(segment_ids), 'icustay_id':icustay_id }
        return sample
    
    def split(self, from_file=True, random_state=0, test=0.2 ):
        n = len(self.data_str)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=random_state)
        if from_file:
            train_icu = pd.read_csv(self.path+'train_icu.csv',index_col=0).iloc[:,0]
            test_icu = pd.read_csv(self.path+'test_icu.csv',index_col=0).iloc[:,0]
            tr_val = np.array(self.index)[self.data_str['icustay_id'].isin(train_icu)]
            test_idx = np.array(self.index)[self.data_str['icustay_id'].isin(test_icu)]
        else:
            tr_val, test_idx = next(sss.split(np.zeros(n), self.data_str.loc[:,'aki_label']))
        tr, val = next(sss.split(np.zeros(len(tr_val)), self.data_str.iloc[tr_val]['aki_label']))     
        tr_idx, val_idx=tr_val[tr], tr_val[val]
        
        tr_set, val_set, te_set, tr_val_set = copy.copy(self), copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.index, val_set.index, te_set.index, tr_val_set.index = tr_idx, val_idx, test_idx, tr_val
        
        return tr_set, val_set, te_set, tr_val_set
        
    