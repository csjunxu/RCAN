## train
# BI, scale 2, 3, 4, 8
##################################################################################################################################
# BI, scale 2, 3, 4, 8
# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
LOG=./../experiment/RCAN_LM_H12W12_AG_BIX2_G10R20P48Ep1000-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN_LM_Avg_Gated --save RCAN_LM_H12W12_AG_BIX2_G10R20P48Ep1000 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --chop --save_results --print_model --epochs 1000 --patch_size 96 --mask_height 12 --mask_width 12 2>&1 | tee $LOG

LOG=./../experiment/RCAN_LM_H24W24_AG_BIX2_G10R20P48Ep1000-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN_LM_Avg_Gated --save RCAN_LM_H24W24_AG_BIX2_G10R20P48Ep1000 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --chop --save_results --print_model --epochs 1000 --patch_size 96 --mask_height 24 --mask_width 24 2>&1 | tee $LOG

LOG=./../experiment/RCAN_LM_H48W48_AG_BIX2_G10R20P48Ep1000-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN_LM_Avg_Gated --save RCAN_LM_H48W48_AG_BIX2_G10R20P48Ep1000 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --chop --save_results --print_model --epochs 1000 --patch_size 96 --mask_height 48 --mask_width 48 2>&1 | tee $LOG

# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
# LOG=./../experiment/RCAN_BIX3_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
# CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG

# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
# LOG=./../experiment/RCAN_BIX4_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
# CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG

# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
# LOG=./../experiment/RCAN_BIX8_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
# CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG
