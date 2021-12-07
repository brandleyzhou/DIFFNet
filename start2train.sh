CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_stereo --height 320 --width 1024 --scheduler_step_size 14  --batch_size 4  --model_name mono_model --png --data_path data_path/kitti
#python train.py --height 320 --width 1024 --scheduler_step_size 14  --batch_size 4  --model_name mono_model --png --data_path data_path/kitti
#python train.py --height 384 --width 1280 --scheduler_step_size 14  --batch_size 4  --model_name mono_model --png --data_path data_path/kitti
