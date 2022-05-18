### for cityscapes evaluation.
python evaluate_depth.py   --load_weights_folder models/diffnet_640x192 --eval_mono --eval_split cityscapes --data_path data_path/cityscapes_preprocessed

### for kitti evaluation.
python evaluate_depth.py   --load_weights_folder models/diffnet_640x192 --eval_mono --eval_split eigen --data_path data_path/kitti
