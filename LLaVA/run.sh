model="/data2/zla/ckpt/llava-v1.5-7b"
dataset="/data2/zla/AdaIAT/coco500"

CUDA_VISIBLE_DEVICES=0 python chair_generate.py --method PAI --alpha 0.5 --name PAI_0.5 --model-path $model --image-folder $dataset
CUDA_VISIBLE_DEVICES=0 python chair_generate.py --method HGAI --alpha 0.5 --name HGAI_0.5 --model-path $model --image-folder $dataset
CUDA_VISIBLE_DEVICES=0 python chair_generate.py --method IAT --alpha 0.8 --name IAT_0.8 --model-path $model --image-folder $dataset
CUDA_VISIBLE_DEVICES=0 python chair_generate.py --method AdaIAT --alpha 6 --beta 0.5 --name AdaIAT_6_0.5 --model-path $model --image-folder $dataset
CUDA_VISIBLE_DEVICES=0 python chair_generate.py --method Greedy --name Greedy_7b --model-path $model --image-folder $dataset