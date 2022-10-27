export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
export TEST_FILE=wikitext-2-raw/wiki.test.raw
export OUTPUT=28_384_jiexiong #pretrain_scratch

CUDA_VISIBLE_DEVICES=3,4,5 python run_language_modeling.py \
    --output_dir=$OUTPUT \
    --model_type=mobilebert \
    --config_name /data/ZLKong/folderfor_Py3.6_2/transformers/examples/question-answering/results/train_squad_mobilebert_jiexiong_28/config.json \
    --do_train \
    --tokenizer_name bert-base-uncased \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --overwrite_output_dir \
    --eval_data_file=$TEST_FILE \
    --mlm  \
    --save_steps=200 \
    --num_train_epochs 1
