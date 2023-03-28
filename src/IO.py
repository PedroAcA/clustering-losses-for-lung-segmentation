import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt


class ImageDataSelfTrain(data.Dataset):
    def __init__(self, config, data_root, data_list):
        self.config = config
        self.sal_root = data_root
        self.sal_source = data_list
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        files_list = self.sal_list[item % self.sal_num].split()
        im_name = files_list[0]
        img_input = load_image(os.path.join(self.sal_root, im_name),  self.config)
        img_output = np.copy(img_input)
        sample = {'input': torch.Tensor(img_input), 'output': torch.Tensor(img_output), 'idx': item, 'im_name': im_name}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, config, data_root, data_list, gt_root):
        self.config = config
        self.sal_root = data_root
        self.sal_source = data_list
        self.gt_root = gt_root
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        files_list = self.sal_list[item % self.sal_num].split()
        im_name = files_list[0]
        gt_name = files_list[1]
        img_input = load_image(os.path.join(self.sal_root, im_name),  self.config)
        img_output = load_image(os.path.join(self.gt_root, gt_name),  self.config)
        sample = {'input': torch.Tensor(img_input), 'output': torch.Tensor(img_output), 'idx': item, 'im_name': im_name}
        return sample

    def __len__(self):
        return self.sal_num


def load_image(path, config):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.convert('L') # convert image to grayscale
    im = im.resize(config.SOLVER.IMG_SIZE)
    in_ = np.array(im, dtype=np.float32)
    if np.max(in_)>1.:
        in_ = in_/255. # image in the range [0,1]

    return in_[np.newaxis, ...]


def get_loader(config, mode='train', test_batch_size=1, pin=False):
    if mode == 'train':
        shuffle = True
        dataset = ImageDataSelfTrain(config, config.DATA.TRAIN.ROOT, config.DATA.TRAIN.LIST)
        batch_size = config.SOLVER.BATCH_SIZE
        drop_last = True
    if mode == 'test':
        shuffle = False
        dataset = ImageDataTest(config, config.DATA.TEST.ROOT, config.DATA.TEST.LIST, config.DATA.TEST.GT_ROOT)
        batch_size = test_batch_size
        drop_last = False

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                  pin_memory=pin, drop_last=drop_last)
    return data_loader


def create_folder(folder):
    os.makedirs(folder, exist_ok=True)


class Checkpoint:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def log_msg(self, msg, filename, mode='a'):
        save_dir = self.config.SAVE_ROOT + self.config.SYSTEM.EXP_NAME + '/'
        create_folder(save_dir)
        with open(save_dir + filename, mode) as f:
            f.write(msg)

    def log_training(self, msg):
        self.log_msg(msg, 'training_log.txt', mode='a')

    def log_testing(self, msg):
        self.log_msg(msg, 'testing_log.txt', mode='a')

    def log_config(self):
        self.log_msg(str(self.config), 'experiment_configuration.yaml', mode='w')

    def save(self):
        save_dir = self.config.SAVE_ROOT + self.config.SYSTEM.EXP_NAME + '/' + 'chkpt/'
        create_folder(save_dir)
        filename = self.config.SYSTEM.EXP_NAME + "_epoch_" + str(self.config.SOLVER.EPOCHS-1) + ".pth"
        print("Saving {}".format(filename))
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))
        print("Saved {}".format(filename))

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    def save_img_tensor(self, img_tensor, filename, title='', grayscale=False, remove_xy_ticks=True, cmap='seismic',
                        alpha=None, dpi=300):
        folder = self.config.SAVE_ROOT + 'results/' + self.config.SYSTEM.EXP_NAME + '/' + 'qualitative/'
        num_of_dims = len(img_tensor.shape)
        assert 2 <= num_of_dims <= 3, "Expected tensor of dimension 2 or 3, but got {}".format(num_of_dims)
        create_folder(folder)
        print("Saving {} to {}".format(filename, folder))
        if grayscale:
            cmap = 'gray'

        if num_of_dims == 2:
            plt.imshow(img_tensor.detach().cpu(), cmap=cmap, alpha=alpha)
        else:
            plt.imshow(img_tensor.detach().cpu().permute(1, 2, 0), cmap=cmap,
                       alpha=alpha)  # permute from C,H,W to H,W,C and show image

        plt.title(title)
        if remove_xy_ticks:
            plt.xticks([])
            plt.yticks([])

        plt.savefig(folder + filename, bbox_inches='tight', dpi=dpi)
        print("Saved {} to {}".format(filename, folder))
        plt.figure()
