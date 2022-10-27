export SQUAD_DIR=data/squad1.0
export CUDA_VISIBLE_DEVICES="1"

python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.2" --master_port=29501 run_squad-alllayer.py \
  --model_type mobilebert \
  --model_name_or_path mrm8488/mobilebert-uncased-finetuned-squadv1 \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --doc_stride 128 \
  --output_dir results/train_squad-alllayer/ \
  --save_steps 10000
