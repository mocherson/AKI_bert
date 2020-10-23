import pickle as pk
from torch.utils.data import Sampler
from collections import Counter
import numpy as np

def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)    
    


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def convert_example_to_feature(example, max_seq_length, tokenizer):

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    # segment id sometimes 0, sometimes 1    
    '''   
    seg_mark = 0
    for i, token in enumerate(tokens):
        reverse = 0
        if token == '|' and i != len(tokens) - 2:
            tokens[i] = '[SEP]'
            reverse = 1
        segment_ids[i] = seg_mark
        if reverse == 1:
            seg_mark = 1 - seg_mark
    '''
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    return input_ids, input_mask, segment_ids


class StratifiedBatchSampler(Sampler):
    """Stratified Sampling
    """
    def __init__(self, labels, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.labels = labels
        self.label_count = Counter(labels)
        self.label_count_per_batch = {k: int(np.round(v*batch_size/len(labels)))  for k, v in self.label_count.items()}
        if 0 in self.label_count_per_batch.values():
            raise ValueError('a class have no samples in a batch, please increase batch size.')
        self.label_ind = {k:np.where(labels==k)[0] for k in self.label_count.keys()}
        self.batch_size=batch_size

    def __iter__(self):
        label_ind_rand = {k:np.random.permutation(v) for k, v in self.label_ind.items()}
        label_ind_rand = {k:label_ind_rand[k][self.label_count[k]%v:].reshape((-1, v)) for k, v in self.label_count_per_batch.items()}
        ind_ext = {k : np.append(v, v[:(len(self)-len(v)),:], axis=0) for k, v in label_ind_rand.items()}
        batchs = np.concatenate(list(ind_ext.values()), axis=1)
        
        return iter(batchs)

    def __len__(self):
        return max([self.label_count[k]//self.label_count_per_batch[k] for k in self.label_count.keys()])
    
class BalancedBatchSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, labels, batch_size, downsample=True):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.labels = labels
        self.downsample = downsample
        self.label_count = Counter(labels)
        self.label_count_per_batch = batch_size//len(self.label_count)
        self.label_ind = {k:np.where(labels==k)[0] for k in self.label_count.keys()}
        self.batch_size=batch_size

    def __iter__(self):
        label_ind_rand={}
        for k, v in self.label_ind.items():
            if self.downsample:
                label_ind_rand[k] = np.random.choice(v,(len(self),self.label_count_per_batch),replace=False)
            else:
                v_ext = np.tile(v,int(np.ceil(len(self)*self.label_count_per_batch/len(v))))
                label_ind_rand[k] = np.random.choice(v_ext,(len(self),self.label_count_per_batch),replace=False)
        
        batchs = np.concatenate(list(label_ind_rand.values()), axis=1)
        
        return iter(batchs)

    def __len__(self):
        return min(self.label_count.values())//self.label_count_per_batch if self.downsample else max(self.label_count.values())//self.label_count_per_batch
    
    
    
    
    
    