#!/bin/bash

# Define seeds
seeds=(42)

# Define configs
configs=(
  "30:32:4:0.00001:0.1:0.00001:8"
  "60:8:6:0.00005:0.3:0.00001:8"
)

echo "========================================="
echo "Sequential Training Pipeline"
echo "========================================="

# Process ONE sequence length at a time
for config in "${configs[@]}"; do
  IFS=':' read -r seq_len emb depth lr dr wd patch <<< "$config"
  datapath="/root/Mine_ROI_Net/country_wise_data/seq_${seq_len}"
  
  echo ""
  echo "========================================="
  echo "PROCESSING SEQ_LEN=$seq_len"
  echo "========================================="
  
  # Run all seeds for THIS sequence length
  for seed in "${seeds[@]}"; do
    echo ""
    echo ">>> Running: SEQ=$seq_len | SEED=$seed | emb=$emb | depth=$depth"
    
    CUDA_VISIBLE_DEVICES=2 python -u TSLANet_classification.py \
      --data_path $datapath \
      --emb_dim $emb \
      --depth $depth \
      --weight_decay $wd \
      --model_id seq${seq_len}_seed${seed}_emb${emb}_d${depth} \
      --load_from_pretrained False \
      --num_epochs 20 \
      --pretrain_epochs 5 \
      --dropout_rate $dr \
      --train_lr $lr \
      --batch_size 64 \
      --patch_size $patch \
      --seed $seed \
      --seq_len $seq_len
    
    if [ $? -ne 0 ]; then
      echo "ERROR: Training failed!"
      exit 1
    fi
  done
  
  echo ""
  echo "✓ Finished all experiments for SEQ_LEN=$seq_len"
  
  # IMPORTANT: Wait a moment between sequence lengths
  # This ensures preprocessing completes properly
  sleep 2
done

echo ""
echo "========================================="
echo "✓ ALL EXPERIMENTS COMPLETED!"
echo "========================================="