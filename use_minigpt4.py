import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


root='/coco/coco_popular.json'

image_path='/val2014/'

sList=[]

res=''

with open(root,'r',encoding='utf-8') as f:
    for jsonObj in f:
        sDict = json.loads(jsonObj)
        sList.append(sDict)

    for s in sList:
        img_ = []
        img=s["image"]
        caption=s["text"]
        labels=s["label"]

        img_list=chat.upload_img(image=os.path.join(image_path,img),img_list=img_)

        llm_message = chat.answer(img_list=img_list,prompt_text=caption)[0]

        print(llm_message)

        res=res+'###'+llm_message+'\n'

with open('/output/coco/pred.txt','w',encoding='utf-8') as fw:
    fw.write(res)
    fw.close()





