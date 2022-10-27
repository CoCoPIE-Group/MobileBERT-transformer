export SQUAD_DIR=data/squad1.0
export PENALTY=squad-penalty_mobilebert-4
export CUDA_VISIBLE_DEVICES="1"

python -m torch.distributed.launch --nproc_per_node=1 run_squad.py \
  --model_type mobilebert \
  --model_name_or_path results/finetune89.7 \
  --do_train \
  --do_eval \
  --rew \
  --do_lower_case \
  --penalty_config_file ${PENALTY} \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --block_row_division 64 \
  --overwrite_output_dir \
  --doc_stride 128 \
  --output_dir results/squad-prune-mobilebert/ \
  --save_steps 100
