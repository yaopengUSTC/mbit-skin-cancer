from .baseset import BaseSet
import cv2
import os


class ISIC(BaseSet):
    def __init__(self, mode='train', cfg=None):
        super().__init__(mode, cfg)

    def _get_image(self, now_info):
        if self.data_type == "jpg":
            fpath = os.path.join(self.data_root, now_info["derm_path"])
            img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class isic_2017(ISIC):
    def __init__(self, mode='train', cfg=None):
        super().__init__(mode, cfg)
        self.dataset_name = "isic_2017"


class isic_2018(ISIC):
    def __init__(self, mode='train', cfg=None):
        super().__init__(mode, cfg)
        self.dataset_name = "isic_2018"


class isic_2019(ISIC):
    def __init__(self, mode='train', cfg=None):
        super().__init__(mode, cfg)
        self.dataset_name = "isic_2019"





