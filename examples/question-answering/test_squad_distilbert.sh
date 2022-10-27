export SQUAD_DIR=data/squad1.0
export CUDA_VISIBLE_DEVICES="1,2,3"

python -m torch.distributed.launch --nproc_per_node=3 run_squad.py \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 3 \
  --per_gpu_eval_batch_size 3 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --overwrite_output_dir \
  --doc_stride 128 \
  --output_dir /tmp/train_squad/ \
  --save_steps 10000
