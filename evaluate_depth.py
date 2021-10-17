from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from  torchvision.utils import save_image
from layers import disp_to_depth
from utils import readlines, sec_to_hm_str
from options import MonodepthOptions
import datasets
import networks
import hr_networks
from viz_map import save_depth, save_visualization,save_error_visualization 
print(torch.__version__)
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
def rank_error(errors, idx = 0, top = 5):
    list_err = []
    rank_list_maxi = []
    rank_list_mini = []
    errors = list(errors)
    for error in errors:
        list_err.append(list(error)[idx])
    copy_list_err = list_err.copy()
    list_err.sort(reverse=True)
    for value in list_err[:top]:
        rank_list_maxi.append(copy_list_err.index(value))
    print("maxi",rank_list_maxi)
    print(list_err[:top])
    for value in list_err[-top:]:
        rank_list_mini.append(copy_list_err.index(value))
    print("mini",rank_list_mini)
    print(list_err[-top:])
    return None
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def visual_error(gt, pred, mask, i):
    """Computation of error metrics between predicted and ground truth depths
    """
    pred_depth_0 = pred
    gt_0 = gt
    pred_depth = pred
    gt_depth = gt
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    ratio = np.median(gt_depth) / np.median(pred_depth)
    pred_depth_0 *= ratio
     
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    pred_depth_0[pred_depth_0 < MIN_DEPTH] = MIN_DEPTH
    pred_depth_0[pred_depth_0 > MAX_DEPTH] = MAX_DEPTH
    gt_0[mask==False] = 1
    abs_rel = np.abs(gt_0 - pred_depth_0) / gt_0
    mask_0 = np.ones(mask.shape)
    mask_0[mask==False] = 0
    mask_0[mask==True]=1
    abs_rel = mask_0 * abs_rel
    save_visualization(abs_rel , i)
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        
        encoder_dict = torch.load(encoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
        decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        ##change encoder and decoder:
        
        #encoder = networks.ResnetEncoder(opt.num_layers, False)
        #encoder = networks.hr_encoder.hrnet32(False)
        
        ## hrnet18
        #encoder = networks.hr_encoder.hrnet18(False)
        encoder = networks.test_hr_encoder.hrnet18(False)
        encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
        ## hrnet64
        #encoder = networks.hr_encoder.hrnet64(False)
        #encoder = networks.test_hr_encoder.hrnet64(False)
        #encoder.num_ch_enc = [ 64, 64, 128, 256, 512]
        #encoder = networks.ResnetEncoder(opt.num_layers, False)
        #encoder = hr_networks.ResnetEncoder(opt.num_layers, opt.weights_init == "pretrained")
        
###################################################################################
        #depth_decoder = hr_networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
        #depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        #depth_decoder = networks.Lite_DepthDecoder(encoder.num_ch_enc)
        model_dict = encoder.state_dict()
        dec_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_model_dict})
        #encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k not in ["height", "width", "use_stereo"]})
        #depth_decoder.load_state_dict(torch.load(decoder_path)) if torch.cuda.is_available() else depth_decoder.load_state_dict(torch.load(decoder_path,map_location = 'cpu'))
        
        encoder.cuda() if torch.cuda.is_available() else encoder.cpu()
        encoder.eval()
        depth_decoder.cuda() if torch.cuda.is_available() else depth_decoder.cpu()
        depth_decoder.eval()
        #pred_disps_viz = []
        pred_disps = []
        print('-->Using\n cuda') if torch.cuda.is_available() else print('-->Using\n CPU')
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            init_time = time.time()
            i = 0 
            for data in dataloader:
                i += 1
                if torch.cuda.is_available():
                     
                    input_color = data[("color", 0, 0)].cuda()
                
                else:
                    input_color = data[("color", 0, 0)].cpu()
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp_0, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp_0.cpu()[:, 0].numpy()
                #pred_disp_viz = pred_disp_0.squeeze()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                #pred_disps_viz.append(pred_disp_viz)
            end_time = time.time()
            inferring = end_time - init_time
            print("===>total time:{}".format(sec_to_hm_str(inferring)))

        pred_disps = np.concatenate(pred_disps)
        #pred_disp_viz = torch.cat(pred_disps_viz)
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
    #gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    
    #tobe_cleaned = [673, 559, 395, 374, 330, 263, 252, 183, 174, 173, 164, 106]
    #hard = list(set([395, 559, 374, 394, 385, 173, 164, 58, 73, 377,395, 559, 548, 374, 385, 477, 388, 394, 73, 518, 395, 73, 374, 549, 394, 106, 388, 330, 260, 68]))
    #hard = list(set([395, 374, 330, 388, 106, 174, 173, 183, 66, 477, 395, 559, 548, 374, 385, 477, 388, 394, 73, 518, 395, 559, 374, 394, 385, 173, 164, 58, 73, 377,395, 394, 374, 173, 559, 152, 330, 174, 64, 183]))
    #hard = list(set([395, 386, 683, 559, 374, 394, 518, 504, 388, 183,395, 559, 374, 394, 385, 173, 164, 58, 73, 377,395, 559, 548, 374, 385, 477, 388, 394, 73, 518, 395, 73, 374, 549, 394, 106, 388, 330, 260, 68]))
    #hard = [395, 386, 683, 559, 374, 394, 518, 504, 388, 183,395, 559, 374, 394, 385, 173, 164, 58, 73, 377,395, 559, 548, 374, 385, 477, 388, 394, 73, 518, 395, 73, 374, 549, 394, 106, 388, 330, 260, 68]
    tobe_cleaned = []
    cleaned = list(range(pred_disps.shape[0]))
    for i in tobe_cleaned:
        if i in cleaned:
            cleaned.remove(i)
    for i in range(pred_disps.shape[0]):
    #for i in cleaned:
    #for i in hard:

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        #pred_depth_viz = 1 / pred_disp_viz[i]
        #print(pred_depth_viz.size())
        #save_error_visualization(gt_depth,pred_depth,i)
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0
        
        #pred_depth_viz *= opt.pred_depth_scale_factor
        #print(pred_depth_viz.size())
        #save_depth(pred_depth_viz,i)

        # for visualize error map
        #visual_error(gt_depth,pred_depth,mask,i)
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    mean_errors = np.array(errors).mean(0)
    ## ranked_error
    ranked_error = rank_error(errors, 0 ,10)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
