### for training on cityscapes
#python train.py --height 416 --width 128  --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed

### for training on kitti
#python train.py --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/kitti
