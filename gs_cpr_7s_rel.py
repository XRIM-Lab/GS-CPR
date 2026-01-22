from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from argparse import ArgumentParser
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs

from tqdm import tqdm
import numpy as np
import math
import cv2
import os
from utils.functions import *

import logging
_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    parser = ArgumentParser(description="GS-CPR_rel for pose estimators")
    parser.add_argument("--pose_estimator", default="dfnet",choices=["ace","dfnet"], type=str)
    parser.add_argument("--scene", default="chess", type=str)
    parser.add_argument("--test_all", action='store_true', default=False)
    args = parser.parse_args()
    original_size = (480, 640)
    pe = args.pose_estimator
    if args.test_all:
        SCENES = ['chess','fire','heads','office','pumpkin','redkitchen','stairs']
    else:
        SCENES = [args.scene]
    
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    log_path = f"./outputs/7scenes/GS_CPR_rel_{pe}_results/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f"Directory {log_path} created.")
    else:
        print(f"Directory {log_path} already exists.")
    for SCENE in tqdm(SCENES):
        logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
        filename= log_path + f'logs_{SCENE}.log',  # 日志文件名
        filemode='w'  # 写入模式，'w' 表示覆盖，'a' 表示追加
        )
        _logger = logging.getLogger(__name__)

        # load_images can take a list of images or a directory
        query_path = f'./datasets/pgt_7scenes_{SCENE}/test/rgb/'
        rendered_path = f'./ACT_Scaffold_GS/data/7scenes/scene_{SCENE}/test/evaluate_{pe}/train_output/render_single_view/'

        predict_pose_w2c_path = f'./coarse_poses/{pe}/7Scenes_pgt/poses_pgt_7scenes_{SCENE}_.txt'
        gt_pose_c2w_path = f'./datasets/pgt_7scenes_{SCENE}/test/poses/'
        gs_depth_path = rendered_path

        gt_pose_c2w_dict = {}
        predict_pose_w2c_dict = {}
        images_list = []
        for filename in os.listdir(query_path):
            if filename.endswith('.png'):
                images_list.append(filename.replace('_frame','/frame'))

        images_list.sort()
        for img_name in images_list:
            pose_file_name = gt_pose_c2w_path + img_name.replace('.color.png','.pose.txt').replace('/frame','_frame')
            c2w_pose = np.loadtxt(pose_file_name)
            gt_pose_c2w_dict[img_name] = c2w_pose
            if pe == 'dfnet':
                predict_w2c_ini= getPredictPos(img_name.replace('-frame','/frame'),predict_pose_w2c_path)
            else:
                predict_w2c_ini= getPredictPos(img_name.replace('/frame','_frame'),predict_pose_w2c_path)
            predict_pose_w2c_dict[img_name] = predict_w2c_ini
            
        
        results_ini = []
        results_final = []
        

        refine_results_path = log_path +  "refine_predictions/" 
        if not os.path.exists(refine_results_path):
            os.makedirs(refine_results_path)
            print(f"Directory {refine_results_path} created.")
        else:
            print(f"Directory {refine_results_path} already exists.")
        ransac_time = 0
        with open(refine_results_path + f'{pe}_refinew2c_mast3r_{SCENE}.txt', 'w') as f:
            for image in tqdm(images_list):  
                try: 
                    image1 = rendered_path + image
                    image2 = query_path + image
                    images = load_images([image1, image2], size=512)
                except:
                    image1 = rendered_path + image.replace('-frame','/frame')
                    image2 = query_path + image
                    images = load_images([image1, image2], size=512)

                pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                output = inference(pairs, model, device, batch_size=batch_size)
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
                poses = scene.get_im_poses()
                P_rel = poses[1].detach().cpu().numpy()
                T_rel = P_rel[:3,3]
                R_rel = P_rel[:3,:3]
                array_to_check = np.array([0, 0, 0])
                P_ini = np.eye(4)
                if np.array_equal(array_to_check, T_rel):
                    P_rel = np.linalg.inv(poses[0].detach().cpu().numpy())
                try:
                    depth_map = np.load(gs_depth_path+image.replace('png','npy').replace('-frame','/frame'))
                except:
                    depth_map = np.load(gs_depth_path+image.replace('png','npy'))
                depth_map_mast3r = output["pred1"]["pts3d"][0, ..., 2].cpu().numpy()
                depth_map_resized = cv2.resize(depth_map, (512, 384), interpolation=cv2.INTER_LINEAR)
                scale_factor = getScale(depth_map_mast3r,depth_map_resized)
                predict_w2c_ini = predict_pose_w2c_dict[image]
                gt_c2w_pose = gt_pose_c2w_dict[image]
                predict_c2w_refine = P_rel.copy()
                predict_c2w_refine[:3,3] = P_rel[:3,:3]@P_ini[:3,3] + scale_factor*P_rel[:3,3]
                
                ini_rot_error,ini_translation_error=cal_campose_error(np.linalg.inv(predict_w2c_ini), gt_c2w_pose)
                results_ini.append([ini_rot_error,ini_translation_error])
                refine_rot_error,refine_translation_error=cal_campose_error(predict_c2w_refine, predict_w2c_ini@gt_c2w_pose)
                results_final.append([refine_rot_error,refine_translation_error])
                combined_list = [image.replace('_frame','/frame')] + rotmat2qvec(np.linalg.inv(predict_c2w_refine)[:3,:3]).tolist() + np.linalg.inv(predict_c2w_refine)[:3,3].tolist()
                output_line = ' '.join(map(str, combined_list))
                f.write(output_line + '\n')
                    
        median_result_ini = np.median(results_ini,axis=0)
        mean_result_ini = np.mean(results_ini,axis=0)

        median_result = np.median(results_final,axis=0)
        mean_result = np.mean(results_final,axis=0)
        pct10_5 = 0
        pct5 = 0
        pct2 = 0
        pct1 = 0
        for err in results_ini:
            r_err = err[0]
            t_err = err[1]
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1
        total_frames = len(results_ini)
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100
        _logger.info('Ini Accuracy:')
        _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
        _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
        _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
        _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

        pct10_5 = 0
        pct5 = 0
        pct2 = 0
        pct1 = 0
        for err in results_final:
            r_err = err[0]
            t_err = err[1]
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1
        total_frames = len(results_final)
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100
        _logger.info('After refine Accuracy:')
        _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
        _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
        _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
        _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

        # standard log
        _logger.info(f"--------------GS-CPR_rel for {pe}:{SCENE}--------------")
        _logger.info("Initial Precision:")
        _logger.info('Median error {}m and {} degrees.'.format(median_result_ini[1], median_result_ini[0]))
        _logger.info('Mean error {}m and {} degrees.'.format(mean_result_ini[1], mean_result_ini[0]))

        _logger.info("After refine Precision:")
        _logger.info('Median error {}m and {} degrees.'.format(median_result[1], median_result[0]))
        _logger.info('Mean error {}m and {} degrees.'.format(mean_result[1], mean_result[0]))



