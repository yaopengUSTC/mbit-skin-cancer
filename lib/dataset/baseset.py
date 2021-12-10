from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
from PIL import Image
import torchvision.transforms as transforms
import lib.data_transform.transforms as extended_transforms
import lib.data_transform.modified_randaugment as rand_augment
import numpy as np


class BaseSet(Dataset):
    def __init__(self, mode="train", cfg=None):
        self.mode = mode
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size
        self.dual_sample = True if cfg.BACKBONE.BBN and mode == "train" else False

        print("Use {} Mode to the network".format(self.color_space))
        self.data_root = cfg.DATASET.ROOT

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = cfg.DATASET.TRAIN_JSON
        elif self.mode == "valid":
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET.VALID_JSON
        elif self.mode == "test":
            print("Loading test data ...", end=" ")
            self.json_path = cfg.DATASET.TEST_JSON
        else:
            raise NotImplementedError

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

        if self.cfg.TRAIN.SAMPLER.TYPE == "oversample" and self.mode == "train":
            self._oversample_data()

        self.val_sample_repeat_num = 0
        if (self.cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE or self.cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE) and self.mode != "train":
            self._order_crop_scale_data()

        random.seed(0)
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode == "train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == 'train':
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.image_transform(img, index)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else -1
        )

        meta = dict()
        if self.dual_sample and self.mode == 'train':
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)

            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_info = self.data[sample_index]
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id']
            sample_img = self.image_transform(sample_img, sample_index)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        return image, image_label, meta

    def _oversample_data(self):
        class_sum = np.array([0] * self.num_classes)
        for anno in self.data:
            category_id = anno["category_id"]
            class_sum[category_id] += 1

        max_num = class_sum.max()
        class_mul = (max_num / class_sum).astype(np.int) - 1  # each class needs to be expanded by multiples: class_mul
        class_rem = max_num % class_sum         # remainder of every class expanded
        dataset_len = len(self.data)
        for i in range(dataset_len):
            class_no = self.data[i]["category_id"]
            if class_mul[class_no] > 0:
                if class_rem[class_no] > 0:
                    self.data.extend([self.data[i]] * (class_mul[class_no] + 1))
                    class_rem[class_no] -= 1
                else:
                    self.data.extend([self.data[i]] * class_mul[class_no])
            else:
                if class_rem[class_no] > 0:
                    self.data.extend([self.data[i]])
                    class_rem[class_no] -= 1

    def _order_crop_scale_data(self):
        if self.cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
            assert self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM in [9, 16, 25, 36], \
                "cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM must be in [9, 16, 25, 36]."
            self.val_sample_repeat_num += self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
        if self.cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE:
            assert self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM <= len(self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME), \
                "cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM must be less than or equal to the length of self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME."
            self.val_sample_repeat_num += self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM

        self.data = np.array(self.data).repeat(self.val_sample_repeat_num).tolist()

    def image_pre_process(self, img):
        if self.cfg.TRAIN.SAMPLER.BORDER_CROP == "pixel":
            if self.cfg.TRAIN.SAMPLER.BORDER_CROP_PIXEL > 0:
                img = img[self.cfg.TRAIN.SAMPLER.BORDER_CROP_PIXEL:-self.cfg.TRAIN.SAMPLER.BORDER_CROP_PIXEL,
                          self.cfg.TRAIN.SAMPLER.BORDER_CROP_PIXEL:-self.cfg.TRAIN.SAMPLER.BORDER_CROP_PIXEL, :]
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img)
            if self.cfg.TRAIN.SAMPLER.BORDER_CROP_RATIO > 0.0:
                sz_0 = int(img.size[0] * (1 - self.cfg.TRAIN.SAMPLER.BORDER_CROP_RATIO))
                sz_1 = int(img.size[1] * (1 - self.cfg.TRAIN.SAMPLER.BORDER_CROP_RATIO))
                crop_method = transforms.CenterCrop((sz_0, sz_1))
                img = crop_method(img)

        if self.cfg.TRAIN.SAMPLER.IMAGE_RESIZE:
            # the short side of the input image resize to a fix size
            resizing = transforms.Resize(self.cfg.TRAIN.SAMPLER.IMAGE_RESIZE_SHORT)
            img = resizing(img)

        if self.cfg.TRAIN.SAMPLER.COLOR_CONSTANCY:
            color_constancy = extended_transforms.ColorConstancy(
                power=self.cfg.TRAIN.SAMPLER.CONSTANCY_POWER,
                gamma=None if self.cfg.TRAIN.SAMPLER.CONSTANCY_GAMMA == 0.0 else self.cfg.TRAIN.SAMPLER.CONSTANCY_GAMMA
            )
            img = color_constancy(img)
        return img

    def image_post_process(self, img):
        # change the format of 'img' to tensor, and change the storage order from 'H x W x C' to 'C x H x W'
        # change the value range of 'img' from [0, 255] to [0.0, 1.0]
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)

        if self.cfg.TRAIN.SAMPLER.FIX_MEAN_VAR.ENABLE:
            normalize = transforms.Normalize(torch.from_numpy(np.array(self.cfg.TRAIN.SAMPLER.FIX_MEAN_VAR.SET_MEAN)),
                                             torch.from_numpy(np.array(self.cfg.TRAIN.SAMPLER.FIX_MEAN_VAR.SET_VAR)))
        else:
            normalize = extended_transforms.NormalizePerImage()
        return normalize(img)

    def image_transform(self, img, index):
        img = self.image_pre_process(img)

        if self.mode == "train":
            img = self._train_transform(img, index)
        else:
            img = self._val_transform(img, index)

        img = self.image_post_process(img)
        return img

    def _train_transform(self, img, index):
        if self.cfg.TRAIN.SAMPLER.AUGMENT.NEED_AUGMENT:  # need data augmentation
            # need another image
            while True:
                rand = np.random.randint(0, len(self.data))
                if rand != index:
                    break
            bg_info = self.data[rand]
            img_bg = self._get_image(bg_info)
            img_bg = self.image_pre_process(img_bg)

            img = self.data_augment_train(img, img_bg)
        else:
            img = self.data_transforms_train(img)

        return img

    def data_augment_train(self, img, img_bg):
        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_bg = torch.from_numpy(np.array(img_bg, dtype=np.uint8))
        blank_replace = tuple([i * 255.0 for i in self.cfg.TRAIN.SAMPLER.FIX_MEAN_VAR.SET_MEAN])
        if self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_METHOD == 'rand':     # RandAugment
            img = rand_augment.distort_image_with_randaugment(img, self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_METHOD,
                                                              self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_LAYER_NUM,
                                                              blank_replace,
                                                              self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_MAG)
        else:       # Modified RandAugment
            img = rand_augment.distort_image_with_modified_randaugment(img, self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_METHOD,
                                                                       img_bg, blank_replace,
                                                                       self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_PROB,
                                                                       self.cfg.TRAIN.SAMPLER.AUGMENT.AUG_MAG)

        # --- random crop ---
        img = Image.fromarray(img.numpy())
        # transforms.RandomCrop(self.input_size),
        crop_method = extended_transforms.RandomCropInRate(nsize=self.input_size,
                                                           rand_rate=(self.cfg.TRAIN.SAMPLER.MULTI_CROP.L_REGION,
                                                                      self.cfg.TRAIN.SAMPLER.MULTI_CROP.S_REGION))
        img = crop_method(img)
        # img.show()
        return img

    def data_transforms_train(self, img):
        # --- random crop ---
        # transforms.RandomCrop(self.input_size),
        crop_method = extended_transforms.RandomCropInRate(nsize=self.input_size,
                                                           rand_rate=(self.cfg.TRAIN.SAMPLER.MULTI_CROP.L_REGION,
                                                                      self.cfg.TRAIN.SAMPLER.MULTI_CROP.S_REGION))
        img = crop_method(img)

        rand_h_flip = transforms.RandomHorizontalFlip()
        rand_v_flip = transforms.RandomVerticalFlip()
        img = rand_h_flip(img)
        img = rand_v_flip(img)

        if not self.cfg.TRAIN.SAMPLER.COLOR_CONSTANCY:
            # Color distortion
            color_distort = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
            img = color_distort(img)
        # img.show()
        return img

    def _val_transform(self, img, index):
        if self.val_sample_repeat_num == 0:     # simple center crop
            crop_method = transforms.CenterCrop(self.input_size)
            img = crop_method(img)
        else:
            idx = index % self.val_sample_repeat_num
            if self.cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE and idx < self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM:   # multi crop
                img = self._val_multi_crop(img, idx)
            else:               # multi scale
                if self.cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
                    idx -= self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
                img = self._val_multi_scale(img, idx)
        # img.show()
        return img

    def _val_multi_crop(self, img, idx):
        img = torch.from_numpy(np.array(img, dtype=np.uint8))
        img_size = img.size()
        num = np.int32(np.sqrt(self.cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM))
        y_n = int(idx / num)
        x_n = idx % num
        if img_size[1] >= img_size[0]:
            x_region = int(img_size[1] * self.cfg.TRAIN.SAMPLER.MULTI_CROP.L_REGION)
            y_region = int(img_size[0] * self.cfg.TRAIN.SAMPLER.MULTI_CROP.S_REGION)
        else:
            x_region = int(img_size[1] * self.cfg.TRAIN.SAMPLER.MULTI_CROP.S_REGION)
            y_region = int(img_size[0] * self.cfg.TRAIN.SAMPLER.MULTI_CROP.L_REGION)
        if x_region < self.input_size[1]:
            x_region = self.input_size[1]
        if y_region < self.input_size[0]:
            y_region = self.input_size[0]
        x_cut = int((img_size[1] - x_region) / 2)
        y_cut = int((img_size[0] - y_region) / 2)

        x_loc = x_cut + int(x_n * (x_region - self.input_size[1]) / (num - 1))
        y_loc = y_cut + int(y_n * (y_region - self.input_size[0]) / (num - 1))
        # Then, apply current crop
        img = img[y_loc:y_loc + self.input_size[0], x_loc:x_loc + self.input_size[1], :]
        img = Image.fromarray(img.numpy())
        return img

    def _val_multi_scale(self, img, idx):
        factor = float(self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx][-3:]) / 100.0 + 1.0
        new_height = round(self.input_size[0] * factor)
        new_width = round(self.input_size[1] * factor)
        crop_method = transforms.CenterCrop((new_height, new_width))
        img = crop_method(img)
        img = img.resize((self.input_size[1], self.input_size[0],), Image.ANTIALIAS)

        if "flip_x" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif "flip_y" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif "rotate_90" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.ROTATE_90)
        elif "rotate_270" in self.cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME[idx]:
            img = img.transpose(Image.ROTATE_270)
        return img

    def get_num_classes(self):
        return self.num_classes

    def get_num_class_list(self):
        class_sum = np.array([0] * self.num_classes)
        if self.mode != "test":
            for anno in self.data:
                category_id = anno["category_id"]
                class_sum[category_id] += 1
        return class_sum.tolist()

    def get_annotations(self):
        return self.data

    def get_image_id_list(self):
        image_id_list = []
        if self.val_sample_repeat_num != 0 and self.mode != "train":
            gap = self.val_sample_repeat_num
        else:
            gap = 1
        for i in range(0, len(self.data), gap):
            image_id = self.data[i]["image_id"]
            image_id_list.append(image_id)

        return image_id_list

    def get_num_images(self):
        if self.val_sample_repeat_num != 0 and self.mode != "train":
            return int(len(self.data) / self.val_sample_repeat_num)
        else:
            return len(self.data)

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        if self.data_type == "jpg":
            fpath = os.path.join(self.data_root, now_info["fpath"])
            img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
