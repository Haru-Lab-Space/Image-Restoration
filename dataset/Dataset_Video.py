import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class DatasetVideo(Dataset):
    def __init__(self, 
        root_path, 
        type_video="blur", 
        num_frames=12,
        patch_size=None,
        noise=None):
        super(DatasetVideo, self).__init__()
        self.num_frames = num_frames
        self.video_list = []
        self.