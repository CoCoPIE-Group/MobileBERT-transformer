export SQUAD_DIR=squad1.1
export CUDA_VISIBLE_DEVICES="1"

python -m torch.distributed.launch --nproc_per_node=1 run_squad-alllayer.py \
  --model_type mobilebert \
  --model_name_or_path google/mobilebert-uncased \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --doc_stride 128 \
  --output_dir results/layerdrop/L-6_H-512_A-8 \
  --save_steps 15000
