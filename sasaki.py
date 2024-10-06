# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GPAT.gpat.detect import detect_and_track
from GPAT.gpat.pose_estimate_from_tracking_data import \
    pose_estimate_from_tracking_data
from GPAT.gpat.utils.gpat2fpat import gpat2fpat

# %%
for video in os.listdir('/mnt/d//sasaki_20240930/video/'):
    video_name = video.split('.')[0]

    # %%
    video_path = f"/mnt/d/sasaki_20240930/video/{video_name}.MP4"
    det_model_path = "/home/ohwada/human_pose_estimation/models/rtmdet_l_8xb32-300e_coco.py"
    det_config_path = "/home/ohwada/human_pose_estimation/models/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    pose_model_path = "/home/ohwada/human_pose_estimation/models/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    pose_config_path = "/home/ohwada/human_pose_estimation/models/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
    output_path = "/mnt/d/sasaki_20240930/" 
    tracking_data_path = f"{output_path}data/{video_name}/tracking_data.csv"

    # %%
    detect_and_track(
        video_path = video_path,
        model_path = det_model_path,
        config_path = det_config_path,
        output_path = output_path,
    )

    # %%
    df = pd.read_csv(tracking_data_path)
    df.head()

    # %%
    unique_ids = df['track_id'].unique()
    max_distance = 0

    for id in unique_ids:
        df_id = df[df['track_id'] == id]
        x1_min = df_id['x1'].min()
        y1_min = df_id['y1'].min()
        x1_max = df_id['x1'].max()
        y1_max = df_id['y1'].max()
        x2_min = df_id['x2'].min()
        y2_min = df_id['y2'].min()
        x2_max = df_id['x2'].max()
        y2_max = df_id['y2'].max()
        distance = np.sqrt((x1_max - x1_min)**2 + (y1_max - y1_min)**2 + (x2_max - x2_min)**2 + (y2_max - y2_min)**2)
        if distance > max_distance:
            max_distance = distance
            selected_id = id
            
    track_id = selected_id
    print(track_id)

    # %%
    pose_estimate_from_tracking_data(
        video_path = video_path,
        tracking_data_path = tracking_data_path,
        model_path = pose_model_path,
        config_path = pose_config_path,
        output_path = output_path,
        track_id = track_id,
    )

    # %%
    gpat2fpat(os.path.join(output_path, "data", video_name))


