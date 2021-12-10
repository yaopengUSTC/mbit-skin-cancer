from .baseset import BaseSet
import cv2
import os


class derm_7pt(BaseSet):
    def __init__(self, mode='train', cfg=None):
        super().__init__(mode, cfg)

    def _get_image(self, now_info):
        if self.data_type == "jpg":
            if self.cfg.TRAIN.SAMPLER.IMAGE_TYPE == "derm":
                fpath = os.path.join(self.data_root, now_info["derm_path"])
            else:
                fpath = os.path.join(self.data_root, now_info["clinic_path"])
            img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
