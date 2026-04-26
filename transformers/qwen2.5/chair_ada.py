import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 指定 GPU 0
import argparse
import random
import json
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import modify_attention_ada

def load_coco_annotations(ann_path):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    img_dict = {}
    category_dict = {c["id"]: c["name"] for c in data["categories"]}

    for img_info in data["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = category_dict[ann["category_id"]]
        img_dict[img_id]["anns"].append(cat_name)

    return img_dict

def find_segment_info(tensor: torch.Tensor, value: int = 151655):
    # 1. 构造布尔掩码
    mask = (tensor == value)
    # 2. 找到所有满足条件的下标
    indices = mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        # 如果根本不存在该值
        return None, None, 0

    # 3. 第一个出现位置
    start = indices[0].item()
    # 4. 最后一个出现位置
    end   = indices[-1].item()
    # 5. 段长（连续或非连续均可用此公式；若只关心连续片段，则需另作处理）
    length = end - start + 1

    return start, end, length

def setup_seeds(seed=1234):
    """固定随机种子以便复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def is_image_file(filename):
    """根据后缀简单判断是否为图片文件，可根据需要扩展"""
    IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMG_EXTS

def main():
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-VL 对文件夹中的图像生成描述，并保存到 JSONL。")
    parser.add_argument("--model-path", type=str, default="ckpt/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL 模型路径或预训练名称，如 'Qwen/Qwen2.5-VL-7B-Instruct' 或本地路径")
    parser.add_argument("--image-folder", type=str, default="COCO2014/coco500",
                        help="待生成描述的图片文件夹路径")
    parser.add_argument('--ann-path', type=str, default="COCO2014/annotations_trainval2014/annotations/instances_val2014.json",
                        help="Path to COCO annotation JSON (e.g. instances_val2014.json)")
    parser.add_argument("--output-file", type=str, default="greedy.jsonl",
                        help="输出 JSONL 文件路径，默认 captions.jsonl")
    parser.add_argument("--prompt", type=str, default="Please describe the picture in detail.",
                        help="用于生成描述的文本提示，可根据需求修改")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="生成时允许的最大新 tokens 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度；若设为 0，则为贪心或 beam-search")
    parser.add_argument("--top_p", type=float, default=None,
                        help="nucleus sampling 的 top_p，若不使用可留 None")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="beam search 的 beam 数，默认为 1")
    parser.add_argument("--device", type=str, default=0,
                        help="指定运行设备，如 'cuda' 或 'cpu'。若不指定，自动选择")
    parser.add_argument("--seed", type=int, default=1234,
                        help="随机种子，默认为 1234")
    args = parser.parse_args()

    setup_seeds(args.seed)

    # 选择设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和 processor
    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",  # 使用自动 device map，根据 GPU 情况放置
    )
    model.to(device)
    model.eval()
    print("Model loaded.")

    print(f"Loading processor from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Processor loaded.")

    # 遍历图片文件
    image_folder = args.image_folder
    if not os.path.isdir(image_folder):
        raise ValueError(f"image-folder '{image_folder}' 不是有效文件夹")
    img_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
    img_files.sort()  # 按文件名排序；若需要按数字排序，可自定义 key

    ############################### ada相关 ###############################
    direction=torch.load("qwen_direction_1w.pt", weights_only=False)
    AT_real=torch.from_numpy(direction["AT_real"]).to("cuda")
    AT_hul=torch.from_numpy(direction["AT_hul"]).to("cuda")
    AT_layers_real=torch.from_numpy(direction["AT_layers_real"]).to("cuda")
    AT_layers_hul=torch.from_numpy(direction["AT_layers_hul"]).to("cuda")

    AT_mul=torch.from_numpy(direction["AT_mul"]).to("cuda")
    AT_diff=torch.from_numpy(direction["AT_diff"]).to("cuda")
    AT_layers_mul=torch.from_numpy(direction["AT_layers_mul"]).to("cuda")
   

    print(type(AT_mul),AT_mul.shape)
    print(type(AT_diff),AT_diff.shape)
    print(type(AT_layers_mul),AT_layers_mul.shape)
    AT_mul=AT_mul-1
    AT_layers_mul=AT_layers_mul-1


    name="ada"

    mul=0.25
    f1=9
    thr1=AT_layers_hul+(AT_layers_real-AT_layers_hul)*mul

    l=4
    r=16
    scope=list(range(l, r))
    modify_attention_ada.llama_head_guide(
        model=model,
        guided_layer_range=(5,18),
        scope=scope,
    )

    output_file=f"result/{name}_{args.max_new_tokens}/{name}_mul={mul}_f1={f1}.jsonl"
    # 打开输出文件
    out_fp = open(output_file, "w", encoding="utf-8")

    for img_file in tqdm(img_files, desc="Generating captions"):
        img_path = os.path.join(image_folder, img_file)

        # 构造 Qwen 消息列表
        # Qwen 要求 messages: 列表，包含 dict: {"type":"image","image":PIL.Image} 和 {"type":"text","text": prompt}
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

        # 构造输入文本（chat template）
        # 注意 apply_chat_template 返回的是不 tokenized 的字符串，需要 tokenize=False
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        # 调用 processor 得到 input_ids、pixel_values 等
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # 移到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start, end, length = find_segment_info(inputs["input_ids"][0])
        input_len=len(inputs["input_ids"][0])
        print(start, end, length)

        tokens_mask=[1, start, end, input_len]


        # 生成
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=args.temperature if args.temperature > 0.0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,

                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,    

                tokens_mask=tokens_mask,
                vtfactor=f1,
                direction=AT_mul,
                thr=thr1,    
            )
        # generated 是 [batch_size, seq_len]，batch_size=1
        # 需要去掉 prompt 部分
        # Qwen2.5-VL API: 先计算 trimmed ids: generated_ids_trimmed = out_ids[len(in_ids):]
        # inputs.input_ids 是一个 batch 张量
        in_ids = inputs["input_ids"]

        # generated shape: [1, L]
        gen_ids = generated["sequences"]
        # 由于 batch_size=1，可直接：
        trimmed = gen_ids[:, in_ids.shape[1]:]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        print(output_text)

        # 获取 image_id
        # image_id = int(img_file.split(".png")[0][-6:])
        image_id = int(img_file.split(".jpg")[0][-6:])

        # 组织输出 JSON 对象
        # 你可以根据自己的需求，提取图片 ID，比如从文件名解析数字；这里示例直接用文件名
        out_obj = {
            "image_id": image_id,
            "caption": output_text,
        }
        # 写入 JSONL
        out_fp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    out_fp.close()
    print(f"Generation finished. Captions saved to {output_file}")

if __name__ == "__main__":
    main()
