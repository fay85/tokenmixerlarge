TF_DETERMINISTIC_OPS=1 \
python train.py \
    --backend musa \
    --lib_path ../tensorflow_musa_extension_2.15/build/libmusa_plugin.so \
    --precision mixed_bf16 \
    --dataset criteo \
    --seed 42 \
    --deterministic_alignment \
    --disable_tf32 \
    --musa_device_index 3 \
    --batch_size 2048 \
    --derive_sparse_embs_from_data \
    --train_size 5000000 \
    --valid_size 500000

