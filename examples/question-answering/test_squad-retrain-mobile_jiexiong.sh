export SQUAD_DIR=data/squad1.0
#export TASK_NAME=WNLI
export PENALTY=squad-penalty_mobilebert-4
export PRUNE_RATIO=squad-mobilebert_prune_ratios_30
export CUDA_VISIBLE_DEVICES="3"
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
python -m torch.distributed.launch --nproc_per_node=1 run_squad.py \
  --model_type mobilebert \
  --model_name_or_path results/squad-prune-mobilebert_jiexiong \
  --do_train \
  --masked_retrain \
  --lr_retrain 5e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --block_row_division 64 \
  --output_dir results/retrain_squad_output_mobilebert_jiexiong \
  --overwrite_output_dir \
  --save_steps 2000
