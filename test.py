import torch
import os
import numpy as np
from utils.model_interaction import load_checkpoint, init_model
import cv2
from dataset.Dataset_Video import Dataset_Video
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.file_interaction import mkdir


def frame_to_videos(frames, save_path):
    frame_size = (256, 256)
    fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

    for frame in frames:
        # frame = np.uint8(frame)
        frame = cv2.resize(frame, frame_size)
        out.write(frame)

    out.release()


def predict(model, path_dir_test, output_video_path=None):
    if output_video_path is None:
        mkdir('predict_video')
    if output_video_path is not None:
        mkdir(output_video_path)

    test_dataset = Dataset_Video(path_dir_test, patch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model.to('cuda')
    model.eval()
    videos = []
    print("Start Testing: \n")
    with torch.no_grad():
        data_iterator = tqdm(test_loader)
        for data in tqdm(data_iterator):
            input_video, target = data['input'], data['output']
            input_video = input_video.to('cuda')
            input_video = input_video.permute(0, 2, 1, 3, 4)
            output = model(input_video)
            output = output.permute(0, 2, 1, 3, 4)
            videos.append(output)
    for video in videos:
        video = video.permute(0, 1, 3, 4, 2)
        frames = video.cpu().numpy()

        video_new = []
        for i, frame in enumerate(frames[0]):
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            # print(frame.shape)
            video_new.append(frame)
        print(len(video_new))
        frame_to_videos(video_new, f'predict_video/video_{i}.mp4')
    print("Done Testing ")
