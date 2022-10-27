export SQUAD_DIR=data/squad1.0
export CUDA_VISIBLE_DEVICES="3"

python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.2" --master_port=29501 run_squad.py \
  --model_type mobilebert \
  --model_name_or_path google/mobilebert-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --local_rank -1 \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --max_answer_length=30 \
  --max_query_length=64 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 4e-5 \
  --evaluate_during_training \
  --logging_steps 2700 \
  --num_train_epochs 1.0 \
  --overwrite_output_dir \
  --output_dir results/train_squad_mobilebert_jiexiong \
  --save_steps 2000    
