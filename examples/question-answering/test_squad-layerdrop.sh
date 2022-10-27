export SQUAD_DIR=data/squad1.0
export CUDA_VISIBLE_DEVICES="1,2,3"
export MODEL_DIR=bert-large-uncased

python -m torch.distributed.launch --nproc_per_node=3 --master_addr="127.0.0.3" --master_port=29502 run_squad_layerdrop.py \
  --model_type bert \
  --model_name_or_path $MODEL_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --doc_stride 128 \
  --output_dir results/bert-large-layerdrop-test \
  --save_steps 15000 \

 
 # --remove_layers \
