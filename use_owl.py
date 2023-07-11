import json
import os

import torch
from PIL import Image

from model_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor,MplugOwlProcessor

pretrained_ckpt='MAGAer13/mplug-owl-llama-7b'

model=MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)

image_processor=MplugOwlImageProcessor.from_pretrained(
    pretrained_ckpt,
)

tokenizer=MplugOwlTokenizer.from_pretrained(
    pretrained_ckpt,
)

processor=MplugOwlProcessor(
    image_processor,
    tokenizer,
)
###准备模型输入

root='/output/coco/coco_popular.json'
image_path='/val2014/'

sList=[]
res_pred=''
with open(root,'r',encoding='utf-8') as f:
    for jsonObj in f:
        sDict = json.loads(jsonObj)
        sList.append(sDict)

    res_text=''
    for s in sList:
        img_ = []
        img=s["image"]
        caption=s["text"]
        labels=s["label"]

        ###get response
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }

        prompts = [
            '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {}'''.format(caption) + '''
        AI: ''']
        print(prompts)
        images = [Image.open(os.path.join(image_path,img))]
        inputs = processor(text=prompts, images=images, return_tensors='pt')

        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = model.generate(**inputs, **generate_kwargs)
        sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        print(sentence)

        res_text=res_text+'##$##'+sentence+'\n'

with open('/coco/pred_m.txt','w',encoding='utf-8') as fw:
    fw.write(res_text)
    fw.close()
