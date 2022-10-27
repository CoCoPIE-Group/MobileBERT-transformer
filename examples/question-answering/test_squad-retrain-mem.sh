export SQUAD_DIR=data/squad1.0
#export TASK_NAME=WNLI
export PENALTY=squad-penalty_bert-4
export PRUNE_RATIO=squad-bert_prune_ratios_0
export CUDA_VISIBLE_DEVICES="4,5,6"
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
python -m torch.distributed.launch --nproc_per_node=3 run_squad_mem.py \
  --model_type bert \
  --model_name_or_path /tmp/debug_squad-prune/ \
  --do_train \
  --masked_retrain \
  --lr_retrain 3e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 3 \
  --per_gpu_eval_batch_size 3 \
  --learning_rate 3e-5 \
  --num_train_epochs 0 \
  --evaluate_during_training \
  --block_row_division 256 \
  --output_dir /tmp/retrain_WNLI_output/ \
  --overwrite_output_dir \
  --logging_steps 10 \
  --save_steps 10000
