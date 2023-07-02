# AKI-BERT
Repository for [AKI-BERT: a Pre-trained Clinical Language Model for Early Prediction of Acute Kidney Injury](url: ) (HealthNLP 2020)



## Download AKI-BERT

The AKI-BERT models can also be downloaded [here](https://northwestern.box.com/s/n2ztlf161lmx81t756xlfmdxz71fpbd8),

`AKI-BC-BERT`, `AKI-BioBERT` and `AKI-baseBERT` are pre-trained from [Bio+Clinical BERT](https://www.aclweb.org/anthology/W19-1909.pdf), [BioBERT](https://arxiv.org/abs/1901.08746), [BERT-base-uncased](https://arxiv.org/pdf/1810.04805.pdf), respectively.


## Reproduce AKI-BERT
#### Pretraining
To reproduce the steps necessary to finetune Bio+Clinical BERT or BioBERT on AKI data, follow the following steps:
1. Run `format_for_bert.py` - Note you'll need to change the file paths at the top of the file.
3. Run `AKI_bert_finetuning.sh`  - Note you'll need to change the TRAIN_FILE as the txt file generated last step; change INITIAL_PATH to the model path from which your want to start the pretraining; change OUT_PATH to the path you want to store the model.


#### AKI early prediction
Assuming your model path is  './AKI_bert_simple_lm_notes/', run 

```
python3 aki_bert.py --gpu 0 --bert_model './AKI_bert_simple_lm_notes/' --do_train --num_train_epochs 5   --dropout 0  --pooling max  --sampling SBS --max_seq_per_doc 180 --max_seq_length 32 
```


## Citation
Please acknowledge the following work in papers or derivative software:

```
@inproceedings{mao2020pre,
  title={A pre-trained clinical language model for acute kidney injury},
  author={Mao, Chengsheng and Yao, Liang and Luo, Yuan},
  booktitle={2020 IEEE International Conference on Healthcare Informatics (ICHI)},
  pages={1--2},
  year={2020},
  organization={IEEE}
}
```
