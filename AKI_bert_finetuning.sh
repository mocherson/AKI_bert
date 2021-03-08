
export TRAIN_FILE=./data/formatted.txt
export INITIAL_PATH=../pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/
# export INITIAL_PATH=../pretrained_bert_tf/biobert_v1.1_pubmed/
# export INITIAL_PATH=bert-base-uncased

CUDA_VISIBLE_DEVICES=9 python ./lm_finetuning/simple_lm_finetuning.py  \
    --train_corpus $TRAIN_FILE   \
    --bert_model $INITIAL_PATH    \
    --do_lower_case     \
    --output_dir ./AKI_orgbert_simple_lm_notes/   \
    --do_train  \
    --on_memory  &
    
CUDA_VISIBLE_DEVICES=0 python3 ./lm_finetuning/pregenerate_training_data.py  \
    --train_corpus $TRAIN_FILE    \
    --bert_model $INITIAL_PATH   \
    --do_lower_case    \
    --output_dir ./data/training_notes/   \
    --epochs_to_generate 3   \
    --max_seq_len 256   
    
CUDA_VISIBLE_DEVICES=8 python ./lm_finetuning/finetune_on_pregenerated.py  \
    --pregenerated_data ./data/training/  \
    --bert_model $INITIAL_PATH  \
    --do_lower_case   \
    --output_dir ./AKI_orgbert_pregenerated_lm_notes/   \
    --epochs 3