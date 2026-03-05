import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image



def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



def init_model(model_path, model_base):
    # 阻止torch的冗余初始化，加速模型初始化
    disable_torch_init()

    # 导入模型
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    # print(tokenizer, model, image_processor, context_len)
    # context_len=2048
    return model_name, tokenizer, model, image_processor, context_len

def insert_img_to_qs(qs, mm_use_im_start_end):
    # <im_start><image><im_end>
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # 如果有图像占位符，则将"<im_start><image><im_end>"或者"<image>"直接替换掉占位符
    if IMAGE_PLACEHOLDER in qs:
        if mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        # 使用开始结尾符，则用"<im_start><image><im_end>/n prompt"
        if mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        # 否则直接用"<image>/n prompt"
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs

def get_conv_mode(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print(conv_mode)
    return conv_mode

def eval_model(args, model_name, tokenizer, model, image_processor, context_len):
    qs = args.query  # 问题prompt
    
    # qs中插入图像，预处理占位符等预定义标识
    qs = insert_img_to_qs(qs, model.config.mm_use_im_start_end)

    conv_mode = get_conv_mode(model_name.lower())  # 获取conversation模板

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # 实例化conversation对象
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> Please describe the picture in detail, you should describe it carefully and answer based on what you actually saw. ASSISTANT:"

    # 读取图像文件并预处理
    image_files = image_parser(args)
    # ['/data2/zla/database/COCO2014/val2014/COCO_val2014_000000581100.jpg']
    images = load_images(image_files)
    # [<PIL.Image.Image image mode=RGB size=640x480 at 0x7F1C9F801C30>]
    image_sizes = [x.size for x in images]
    # [(640, 480)]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # torch.Size([1, 3, 336, 336])
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    # input_ids:
    # tensor([[    1,   319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116,
    #          21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
    #            322,  1248,   568,  6089,   304,   278,  1404, 29915, 29879,  5155,
    #          29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
    #            278,  7623,   297,  9493, 29892,   366,   881,  8453,   372, 16112,
    #            322,  1234,  2729,   373,   825,   366,  2869,  4446, 29889,   319,
    #           1799,  9047, 13566, 29901]], device='cuda:0')
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        # output_ids: torch.Size([1, 130])
        # tensor([[    1,   450,  1967,  5680,   263,  2318,   310,   330,  3055,   600,
        #         267,   322,   274,  1242,   297,   263, 17455, 29891,  1746, 29889,
        #         1670,   526,  2211,   330,  3055,   600,   267,   297,   278,  9088,
        #         29892,   411,   697, 13407, 17649,   304,   278,  2175,  2625, 29892,
        #         1790,   297,   278,  7256, 29892,   322,   278,  4654,   697,   373,
        #         278,  1492,  2625, 29889,  2688,   526,   599, 13407,  2978,   263,
        #         5447, 29892, 10075, 25738,   528,  1943, 29889,    13,    13,   797,
        #         6124,   304,   278,   330,  3055,   600,   267, 29892,   727,   526,
        #         1023,   274,  1242,   297,   278,  1746, 29889,  3118, 21282,   338,
        #         5982,  2978,   278,  2175,  2625,   310,   278,  1967, 29892,  1550,
        #         278,   916, 21282,   338,   373,   278,  1492,  2625, 29889,   450,
        #         274,  1242,  2615,   304,   367,  2646, 19583,   373,   278, 17455,
        #         29892,  4969,   263, 10776,  1319,   322,  5613,  9088, 29889,     2]])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    # conv.dict(): {'system': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", 'roles': ('USER', 'ASSISTANT'), 'messages': [['USER', '<image>\nPlease describe the picture in detail, you should describe it carefully and answer based on what you actually saw.'], ['ASSISTANT', None]], 'offset': 0, 'sep': ' ', 'sep2': '</s>'}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
