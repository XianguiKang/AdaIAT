import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import json

import methods

# CUDA_VISIBLE_DEVICES=1 python -m research.chair_adaAmplify3_1
# CUDA_VISIBLE_DEVICES=6 python -m research.chair_adaAmplify3_1 --image-folder /home/amax/check/zla/hallucination2/image_ori_2/image
# CUDA_VISIBLE_DEVICES=1 python chair_generate.py --method IAT --alpha 0.8 --name IAT_0.8 



########################### 工具函数 ###########################
def setup_seeds():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, img_files, tokenizer, image_processor, model_config, prompt, image_folder):
        self.img_files = img_files
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt
        self.image_folder=image_folder

    def __getitem__(self, index):

        image_file = self.img_files[index]
        qs = self.prompt
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, qs, image_file

    def __len__(self):
        return len(img_files)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, qs, image_file= zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, qs, image_file


# DataLoader
def create_data_loader(img_files, tokenizer, image_processor, model_config, prompt, image_folder, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(img_files, tokenizer, image_processor, model_config, prompt, image_folder=image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader



########################### 配置不同方法 ###########################



def intervene(model, method, alpha, split, scope, direction, thr):
    methods_args = {}
    if method == "Greedy" or method == "Sample":
        pass
    elif method == "PAI":
        methods.PAI.llama_head_guide(
            model=model,
            guided_layer_range=(2,32),
            alpha=alpha,
            img_start_idx=split[1],
            img_end_idx=split[2]
        )
    elif method == "HGAI":
        methods.HGAI.llama_head_guide(
            model=model,
            guided_layer_range=(5,18),
            aggregation="mean",
            alpha=alpha,
            img_start_idx=split[1],
            img_end_idx=split[2]
        )
    elif method == "IAT":
        methods.IAT.llama_head_guide(
            model=model,
            guided_layer_range=(0,31),
            scope=scope,
        )
        methods_args["split"] = split
        methods_args["alpha"] = alpha
    elif method == "AdaIAT":
        methods.AdaIAT.llama_head_guide(
            model=model,
            guided_layer_range=(0,31),
            scope=scope,
        )
        methods_args["split"] = split
        methods_args["alpha"] = alpha
        methods_args["direction"] = direction
        methods_args["thr"] = thr
        
    return methods_args

def InitAdaIAT(args, beta):
    param = args.model_path.split('/')[-1]
    if param=="llava-v1.5-7b":
        data=torch.load("llava_direction_1w.pt")
    else:  # 用于其他模型自行设置
        pass

    AT_layers_real=torch.from_numpy(data["AT_layers_real"]).to("cuda")
    AT_layers_hul=torch.from_numpy(data["AT_layers_hul"]).to("cuda")
    AT_mul=torch.from_numpy(data["AT_mul"]).to("cuda")
    AT_layers_mul=torch.from_numpy(data["AT_layers_mul"]).to("cuda")

    print(type(AT_mul),AT_mul.shape)
    print(type(AT_layers_mul),AT_layers_mul.shape)

    direction=AT_mul-1
    thr=AT_layers_hul+(AT_layers_real-AT_layers_hul)*beta
    return direction, thr

# python chair_generate.py --method AdaIAT --alpha 6 --beta 0.5 --name AdaIAT_6_0.5 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="/home/amax/check/zla/ckpt/ckpt/llava-v1.5-7b")  # 选择的llava模型参数
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--name", type=str, default="ada3_1_7b")  # 实验名字
    parser.add_argument("--image-folder", type=str, default="/home/amax/check/zla/database/COCO2014/coco500")  # 数据集路径
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--method", type=str, choices=['Greedy', 'Sample', 'PAI', 'HGAI', 'IAT', 'AdaIAT'], default='AdaIAT')  # 使用的方法名
    parser.add_argument("--max_new_tokens", type=int, default=512)  # 最大tokens数
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.5)

    
    args = parser.parse_args()

    name=args.name  # 只增幅图像和文本
    

    # ========================================
    #             Model Initialization
    # ========================================
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    img_files = os.listdir(args.image_folder)
    img_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))


    # 封装数据集
    prompt="Please describe the picture in detail."

    data_loader = create_data_loader(img_files, tokenizer, image_processor, model.config, prompt, image_folder=args.image_folder)

    base_dir  = f"result/chair/{name}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # 干预方法初始化
    method = args.method
    alpha = args.alpha
    split=[1, 35, 611, 625]  # 划分出五个区域：BOS, 图前prompt，图，图后prompt，生成文本
    scope=list(range(5,18))
    direction, thr = InitAdaIAT(args, args.beta)
    methods_args = intervene(model, method, alpha=alpha, split=split, scope=scope, direction=direction, thr=thr)
    
    index=0
    for (input_ids, image_tensor, image_sizes, qs, img_file) in tqdm(data_loader):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        img_file=img_file[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                **methods_args
                )

            
        output_ids=output_ids.sequences
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()        
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_save = {}
        img_save["image_id"] = img_id
        img_save["caption"] = outputs

        with open(os.path.join(base_dir, f'{name}.jsonl'), "a") as f:
            json.dump(img_save, f)
            f.write('\n')
        index+=1
    


