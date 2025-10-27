#!/bin/bash
mkdir -p logs

COMMON_ARGS="--batch_size 32 --epochs 100 --lr 0.001 --eval_interval 5 --use_wandb"
SRC="bahamas"
TGT="darkskies"

echo "=== BASELINE ==="
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1  --weighting_scheme none --pretrained --run_name baseline_pre_squeezenet1_1 | tee logs/baseline_pre_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1  --weighting_scheme none --run_name baseline_scratch_squeezenet1_1 | tee logs/baseline_scratch_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18   --weighting_scheme none --run_name baseline_scratch_resnet18 | tee logs/baseline_scratch_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18  --weighting_scheme none --pretrained --run_name baseline_pre_resnet18 | tee logs/baseline_pre_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model small_cnn   --weighting_scheme none --run_name baseline_scratch_small_cnn | tee logs/baseline_scratch_small_cnn.log
echo "=== CLASS-WEIGHTING SCHEMES ==="
echo "--- Weighting: inverse_frequency ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1  --weighting_scheme inverse_frequency --run_name w_inverse_frequency_scratch_squeezenet1_1 | tee logs/w_inverse_frequency_scratch_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --weighting_scheme inverse_frequency --run_name w_inverse_frequency_pre_squeezenet1_1 | tee logs/w_inverse_frequency_pre_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18  --weighting_scheme inverse_frequency --run_name w_inverse_frequency_scratch_resnet18 | tee logs/w_inverse_frequency_scratch_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18 --pretrained --weighting_scheme inverse_frequency --run_name w_inverse_frequency_pre_resnet18 | tee logs/w_inverse_frequency_pre_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model small_cnn  --weighting_scheme inverse_frequency --run_name w_inverse_frequency_scratch_small_cnn | tee logs/w_inverse_frequency_scratch_small_cnn.log
echo "--- Weighting: sqrt_inverse ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1  --weighting_scheme sqrt_inverse --run_name w_sqrt_inverse_scratch_squeezenet1_1 | tee logs/w_sqrt_inverse_scratch_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --weighting_scheme sqrt_inverse --run_name w_sqrt_inverse_pre_squeezenet1_1 | tee logs/w_sqrt_inverse_pre_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18  --weighting_scheme sqrt_inverse --run_name w_sqrt_inverse_scratch_resnet18 | tee logs/w_sqrt_inverse_scratch_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model resnet18 --pretrained --weighting_scheme sqrt_inverse --run_name w_sqrt_inverse_pre_resnet18 | tee logs/w_sqrt_inverse_pre_resnet18.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model small_cnn  --weighting_scheme sqrt_inverse --run_name w_sqrt_inverse_scratch_small_cnn | tee logs/w_sqrt_inverse_scratch_small_cnn.log


echo "=== MIXUP STRATEGIES ==="
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --use_mixup --mixup_strategy random --weighting_scheme inverse_frequency --run_name mixup_random_pre_squeezenet1_1 | tee logs/mixup_random_pre_squeezenet1_1.log
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --use_mixup --mixup_strategy same_index  --weighting_scheme inverse_frequency --run_name mixup_same_index_pre_squeezenet1_1 | tee logs/mixup_same_index_pre_squeezenet1_1.log


echo "=== ADAPTATION METHODS ==="
echo "--- Adaptation: dann ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation dann  --weighting_scheme inverse_frequency --run_name adapt_dann_pre_squeezenet1_1 | tee logs/adapt_dann_pre_squeezenet1_1.log
echo "--- Adaptation: cdan ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation cdan  --weighting_scheme inverse_frequency --run_name adapt_cdan_pre_squeezenet1_1 | tee logs/adapt_cdan_pre_squeezenet1_1.log
echo "--- Adaptation: mmd ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation mmd  --weighting_scheme inverse_frequency --run_name adapt_mmd_pre_squeezenet1_1 --adaptation_weight 1.0 | tee logs/adapt_mmd_pre_squeezenet1_1.log
echo "--- Adaptation: coral ---"
python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation coral  --weighting_scheme inverse_frequency --run_name adapt_coral_pre_squeezenet1_1 --adaptation_weight 10 | tee logs/adapt_coral_pre_squeezenet1_1.log


#echo "=== BEST COMBO ADAPTATION METHODS ==="
#echo "--- Adaptation: cdan ---"
#python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation cdan --use_mixup --mixup_strategy same_index  --weighting_scheme inverse_frequency --run_name adapt_cdan_pre_squeezenet1_1 | tee logs/adapt_cdan_mixup_same_pre_squeezenet1_1.log
#python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation cdan --use_mixup --mixup_strategy random  --weighting_scheme inverse_frequency --run_name adapt_cdan_mixup_random_pre_squeezenet1_1 | tee logs/adapt_cdan_mixup_random_pre_squeezenet1_1.log
echo "--- Adaptation: mmd ---"
#python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation mmd --use_mixup --mixup_strategy same_index --weighting_scheme inverse_frequency --run_name adapt_mmd_mixup_same_pre_squeezenet1_1 --adaptation_weight 1.0 | tee logs/adapt_mmd_mixup_same_pre_squeezenet1_1.log
#python main.py $COMMON_ARGS --source_domain $SRC --target_domain $TGT --model squeezenet1_1 --pretrained --adaptation mmd --use_mixup --mixup_strategy random --weighting_scheme inverse_frequency --run_name adapt_mmd_mixup_random_pre_squeezenet1_1 --adaptation_weight 1.0 | tee logs/adapt_mmd_mixup_random_pre_squeezenet1_1.log

echo "=== Upper Bound ==="
#python main.py $COMMON_ARGS --source_domain darkskies --target_domain bahamas --model squeezenet1_1  --weighting_scheme inverse_frequency --pretrained --run_name target_pre_squeezenet1_1 | tee logs/target_pre_squeezenet1_1.log