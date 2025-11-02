from .ade20k_gpt2 import ADE20K as ADE20K_GPT
from .ade20k2 import ADE20K
from .city_scapes import CS
from .coco_object import COCOOBJ
from .coco_stuff import COCOSTUFF
from .context60 import PC
from .voc21 import VOC

# def load_CuPL_templates(dataset_name):
#     dname = dataset_name.lower()
#     if dname == "caltech101":
#         return CALTECH101_TEMPLATES
#     elif dname == "oxfordpets":
#         return OXFORD_PETS_TEMPLATES
#     elif dname == "stanfordcars":
#         return STANFORD_CARS_TEMPLATES
#     elif dname == "oxfordflowers":
#         return OXFORD_FLOWERS_TEMPLATES
#     elif dname == "food101":
#         return FOOD101_TEMPLATES
#     elif dname == "fgvcaircraft":
#         return FGVC_AIRCRAFT_TEMPLATES
#     elif dname == "describabletextures":
#         return DTD_TEMPLATES
#     elif dname == "eurosat":
#         return EUROSAT_TEMPLATES
#     elif dname == "sun397":
#         return SUN397_TEMPLATES
#     elif dname == "ucf101":
#         return UCF101_TEMPLATES
#     elif "imagenet" in dname:
#         return IMAGENET_TEMPLATES