
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


'''
    Dataset for DLPv2
'''
class CalvinDatasetImage(Dataset):
    def __init__(self, mode, data_path=None, image_size=128, ep_len=100, sample_length=20):
        assert mode in ['training', 'validation', 'val', 'train', 'valid']
        if mode == 'valid':
            mode = 'validation'
        if mode == 'val':
            mode = 'validation'
        if mode == 'train':
            mode = 'training'

        self.data_path = os.path.join(data_path, mode)
        self.data_files = [f for f in os.listdir(self.data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor() 
        ])
    
    def __len__(self):
        return len(self.data_files)
    

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        episode_path = os.path.join(self.data_path, file_name)
        data = np.load(episode_path, allow_pickle=True)
        img_static = data["rgb_static"]
        img_static = self.transform(img_static)
        imgs = []
        imgs.append(img_static)
        imgs = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0) 
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        # return img_static, pos, size, id, in_camera
        return imgs, pos, size, id, in_camera


'''
    Dataset for DDLP
'''
class CalvinDataset(Dataset):
    def __init__(self, mode, data_path, ep_len=20, sample_length=64, image_size=128, transform=None):

        assert mode in ['training', 'validation', 'val', 'train', 'valid']
        if mode == 'valid':
            mode = 'validation'
        if mode == 'val':
            mode = 'validation'
        if mode == 'train':
            mode = 'training'

        self.caption_path = f'{os.path.join(data_path, mode)}/lang_annotations/auto_lang_ann.npy'
        self.sample_length = sample_length
        self.data_path = os.path.join(data_path, mode)
        self.caption_data = self.load_caption_data(self.caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor() 
        ])

    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        return annotations

    def __getitem__(self, index):
        annotation = self.caption_data[index]
        start_epi = annotation[0][0]

        images = []
        for i in range(0, self.sample_length):
            epi_num = str(start_epi + (i*2)).zfill(7)
            file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
            data = np.load(file_path)
            img = data["rgb_static"]
            img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=0)
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        return images, pos, size, id, in_camera
    
    def __len__(self):
        return len(self.caption_data)
    

'''
    Dataset for DLPv2 / includes tasks (one-hot-encoded)
    model still needs to be adjusted
'''
class CalvinDatasetTasks(Dataset):
    def __init__(self, mode, data_path, ep_len=20, sample_length=64, image_size=128, transform=None):
        
        tasks = [
            "rotate_red_block_right",
            "rotate_red_block_left",
            "rotate_blue_block_right",
            "rotate_blue_block_left",
            "rotate_pink_block_right",
            "rotate_pink_block_left",
            "push_red_block_right",
            "push_red_block_left",
            "push_blue_block_right",
            "push_blue_block_left",
            "push_pink_block_right",
            "push_pink_block_left",
            "move_slider_left",
            "move_slider_right",
            "open_drawer",
            "close_drawer",
            "lift_red_block_table",
            "lift_blue_block_table",
            "lift_pink_block_table",
            "lift_red_block_slider",
            "lift_blue_block_slider",
            "lift_pink_block_slider",
            "lift_red_block_drawer",
            "lift_blue_block_drawer",
            "lift_pink_block_drawer",
            "place_in_slider",
            "place_in_drawer",
            "push_into_drawer",
            "stack_block",
            "unstack_block",
            "turn_on_lightbulb",
            "turn_off_lightbulb",
            "turn_on_led",
            "turn_off_led",
        ]
        self.task_to_idx = {task: idx for idx, task in enumerate(tasks)}

        assert mode in ['training', 'validation', 'val', 'train', 'valid']
        if mode == 'valid':
            mode = 'validation'
        if mode == 'val':
            mode = 'validation'
        if mode == 'train':
            mode = 'training'

        self.caption_path = f'{os.path.join(data_path, mode)}/lang_annotations/auto_lang_ann.npy'
        self.sample_length = sample_length
        self.data_path = os.path.join(data_path, mode)
        self.caption_data = self.load_caption_data(self.caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor() 
        ])

    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["task"]))
        return annotations
    
    def one_hot_encode_task(self, task):
        idx = self.task_to_idx[task]
        one_hot_task = torch.zeros(len(self.task_to_idx))
        one_hot_task[idx] = 1
        return one_hot_task
    
    def __getitem__(self, index):
        annotation = self.caption_data[index]
        start_epi = annotation[0][0]
        start_epi = annotation[0][0]
        task = annotation[1]
        one_hot_task = self.one_hot_encode_task(task)

        images = []
        for i in range(0, self.sample_length):
            epi_num = str(start_epi + (i*2)).zfill(7)
            file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
            data = np.load(file_path)
            img = data["rgb_static"]
            img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=0)
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        return images, pos, size, id, in_camera, one_hot_task
    
    def __len__(self):
        return len(self.caption_data)