## Single Model Deep Learning on Imbalanced Small Datasets for Skin Lesion Classification
Peng Yao, Shuwei Shen, Mengjuan Xu, Peng Liu, Fan Zhang, Jinyu Xing, Pengfei Shao, Benjamin Kaffenberger, and Ronald X. Xu*

This repository is the official PyTorch implementation of paper [Single Model Deep Learning on Imbalanced Small Datasets for Skin Lesion Classification](https://arxiv.org/abs/2102.01284). 

## Main requirements

  * **torch == 1.5.1**
  * **Python 3**

## Environmental settings
This repository is developed using python **3.6** on Ubuntu **16.04 LTS**. The CUDA version is **9.2**. For all experiments, we use **two NVIDIA 2080ti GPU card** for training and testing. 

## Usage
```bash
# To train ISIC 2018:
python main/train.py  --cfg configs/isic_2018.yaml

# To validate with the best model:
python main/valid.py  --cfg configs/isic_2018.yaml
```

You can change the experimental setting by simply modifying the parameter in the yaml file.

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `category_id`, `derm_height`, `derm_width` and `derm_path`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': ISIC_0031633,
                        'category_id':4,
                        'derm_height':450,
                        'derm_width':600,
                        'derm_path': '/work/image/skin_cancer/HAM10000/ISIC_0031633.jpg'
                    },
                    ...
                   ]
    'num_classes':7
}
```
You can use the following code to convert from the original format of Derm7PT, ISIC 2017, ISIC 2018 or ISIC 2019. 
The images and annotations of ISIC can be downloaded at [ISIC](https://challenge.isic-archive.com/data/).
The images and annotations of Derm7PT can be downloaded at [Derm7PT](https://derm.cs.sfu.ca).

```bash
# Convert from the original format of ISIC 2017
python tools/convert_from_ISIC_2017.py --file ISIC_2017.csv --root /work/image/skin_cancer/ISIC_2017 --sp /work/skin_cancer/jsons
```

## Contacts
If you have any questions about our work, please do not hesitate to contact us by email.

Peng Yao: yaopeng@ustc.edu.cn

