import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import os
import torch
import cv2
import random
from PIL import Image


class Dataset_Video(Dataset):
    def __init__(self, 
        root_path, 
        input_video_type="blur",
        output_video_type="gt",
        num_frames=12,
        patch_size=None,
        noise=None):
        super(Dataset_Video, self).__init__()
        self.num_frames = num_frames
        
        # Load image path
        self.input_video_list = []
        self.output_video_list = []
        for v in os.listdir(f"{root_path}/{output_video_type}"):
            self.input_video_list.append(f"{root_path}/{input_video_type}/{v}")
            self.output_video_list.append(f"{root_path}/{output_video_type}/{v}")
        
        self.patch_size = patch_size
        self.size = len(self.output_video_list)
        self.noise = noise
    
    def __len__(self):
        return self.size
    
    def _inject_noise(self, img, noise):
        noise = (noise**0.5) * torch.rand()
        out = img + noise
        return out.clamp(0, 1)

    def _fetch_chunk(self, index):
        input_video = cv2.VideoCapture(self.input_video_list[index])
        output_video = cv2.VideoCapture(self.output_video_list[index])
        height, width = int(input_video.get(4)), int(output_video.get(3))
        total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            ValueError(f"No enough frame in video {total_frames} for default {self.num_frames}.")
        
        """
            If total frame is larger than default.
            We just get self.num_frames at the end.
        """
        start_frame_id = random.randint(0, total_frames - self.num_frames)

        # Load frame from video
        input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        output_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        input_imgs = [input_video.read()[1] for i in range(self.num_frames)]
        output_imgs = [output_video.read()[1] for i in range(self.num_frames)]

        if input_imgs[0] is None:
            ValueError(f"Frame in image at {index} - {self.output_video_list[index]} is not exist.")

        input_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in input_imgs]
        output_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in output_imgs]
        
        pad_width = self.patch_size - width if width < self.patch_size else 0
        pad_height = self.patch_size - height if height < self.patch_size else 0

        if pad_width != 0 or pad_height != 0:
            input_imgs = [F.pad(img, (0, 0, pad_width, pad_height), padding_mode='reflect') for img in input_imgs]
            output_imgs = [F.pad(img, (0, 0, pad_width, pad_height), padding_mode='reflect') for img in output_imgs]

        aug = random.randint(0, 2)
        if aug == 1:
            input_imgs = [F.adjust_gamma(img, 1) for img in input_imgs]
            output_imgs = [F.adjust_gamma(img, 1) for img in output_imgs]

        enlarge_factor = random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = self.patch_size * enlarge_factor
        crop_size = min(height, width, crop_size)
        height_crop = int(crop_size * random.uniform(1.1, 0.9))
        width_crop = int(crop_size * random.uniform(1.1, 0.9))
        height_crop = min(height_crop, height)
        width_crop = min(width_crop, height)

        rr = random.randint(0, height - height_crop)
        cc = random.randint(0, width - width_crop)

        # Crop match
        input_imgs = [F.resize(img.crop((cc, rr, cc+width_crop, rr+height_crop)), (self.patch_size, self.patch_size)) for img in input_imgs]
        output_imgs = [F.resize(img.crop((cc, rr, cc+width_crop, rr+height_crop)), (self.patch_size, self.patch_size)) for img in output_imgs]

        input_imgs = [F.to_tensor(img) for img in input_imgs]
        output_imgs = [F.to_tensor(img) for img in output_imgs]

        if self.noise:
            noise_level =  self.noise * random.random()
            input_imgs = [self._inject_noise(img, noise_level) for img in input_imgs]
            output_imgs = [self._inject_noise(img, noise_level) for img in output_imgs]

        aug = random.randint(0, 8)

        # Data augmentation
        if aug == 1:
            input_imgs = [img.flip(1) for img in input_imgs]
            output_imgs = [img.flip(1) for img in output_imgs]
        elif aug == 2:
            input_imgs = [img.flip(2) for img in input_imgs]
            output_imgs = [img.flip(2) for img in output_imgs]
        elif aug == 3:
            input_imgs = [torch.rot90(img, dims=(1,2)) for img in input_imgs]
            output_imgs = [torch.rot90(img, dims=(1,2)) for img in output_imgs]
        elif aug==4:
            input_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in input_imgs]
            output_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in output_imgs]
        elif aug==5:
            input_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in input_imgs]
            output_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in output_imgs]
        elif aug==6:
            input_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in input_imgs]
            output_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in output_imgs]
        elif aug==7:
            input_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in input_imgs]
            output_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in output_imgs]

        return input_imgs, output_imgs
    
    def __getitem__(self, index):
        index_ = index % self.size
        input_imgs, output_imgs = self._fetch_chunk(index_)
        return {
            "input": torch.stack(input_imgs, dim = 0),
            "output": torch.stack(output_imgs, dim = 0)
            }