import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def amplify(t, f):
    return t+t.abs()*f
    # return t+t.abs().mean(dim=1, keepdim=True)*f

# 将视觉的注意力随机打乱
def random_shuffle(slice_part):
    slice_part_flat = slice_part.view(-1, slice_part.size(-1))  # 取最后一个维度进行shuffle
    indices = torch.randperm(slice_part_flat.size(0))
    shuffled_slice_part_flat = slice_part_flat[indices]
    shuffled_slice_part = shuffled_slice_part_flat.view(slice_part.shape)
    return shuffled_slice_part

def swap_attention_values(x, a, b):
    # 提取目标切片 [:, :, -1, a:b]
    x_slice = x[:, :, -1, a:b]  # 形状为 [1, 32, 1, (b-a)]
    original_shape = x_slice.shape
    
    # 展平处理
    x_flat = x_slice.flatten()
    n = x_flat.numel()
    k = int(0.2 * n)
    
    if k == 0:
        return x  # 无需交换
    
    # 找到最大和最小的k个元素的索引
    max_vals, max_indices = torch.topk(x_flat, k, largest=True)
    min_vals, min_indices = torch.topk(x_flat, k, largest=False)
    
    # 复制原始值以避免覆盖问题
    original_max = max_vals.clone()
    original_min = min_vals.clone()
    
    # 交换值
    x_flat[max_indices] = original_min
    x_flat[min_indices] = original_max
    
    # 将修改后的数据恢复形状并赋值回原张量
    x[:, :, -1, a:b] = x_flat.view(original_shape)
    
    return x


def shapen_attention(x, a, b):
    # 提取目标切片 [:, :, -1, a:b]
    x_slice = x[:, :, -1, a:b]  # 形状为 [1, 32, 1, (b-a)]
    original_shape = x_slice.shape
    
    # 展平处理
    x_flat = x_slice.flatten()
    n = x_flat.numel()
    k = int(0.1 * n)
    
    if k == 0:
        return x  # 无需交换
    
    # 找到最大和最小的k个元素的索引
    max_vals, max_indices = torch.topk(x_flat, k, largest=True)
    min_vals, min_indices = torch.topk(x_flat, k, largest=False)
    
    # 复制原始值以避免覆盖问题
    original_max = max_vals.clone()
    original_min = min_vals.clone()
    
    # 交换值
    x_flat[max_indices] = original_max*2
    x_flat[min_indices] = original_min*0.5
    
    # 将修改后的数据恢复形状并赋值回原张量
    x[:, :, -1, a:b] = x_flat.view(original_shape)
    
    return x

def mask_max(x, t1, t2, factor):

    x = x.clone()
    target_slice = x[..., t1:t2]
    L = target_slice.size(-1)

    # 计算需要置零的元素数量（向上取整）
    k = math.ceil(L * factor)
    k = min(max(k, 1), L)  # 确保k在[1, L]范围内

    # 批量处理所有切片
    flat_view = target_slice.view(-1, L)
    _, indices = torch.topk(flat_view, k=k, dim=1, largest=True)
    
    # 创建掩码并置零
    mask = torch.zeros_like(flat_view, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    flat_view[mask] = -100
    
    # 恢复原始形状
    x[..., t1:t2] = flat_view.view_as(target_slice)
    return x



def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    ### vision attention modification
    split = kwargs.get('split', None)
    alpha = kwargs.get('alpha', None)
    if split is not None and self.layer_idx in self.scope:
        attn_weights[:, :, -1, split[3]:]=amplify(attn_weights[:, :, -1, split[3]:], alpha)        
        # attn_weights[:, :, -1, split[2]:split[3]]=amplify(attn_weights[:, :, -1, split[2]:split[3]], alpha)        


    ### vision attention modification

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_head_guide(model, guided_layer_range, scope):
    layer_list = guided_layer_range if len(guided_layer_range) == 1 else list(range(guided_layer_range[0], guided_layer_range[1]))

    for i in layer_list:
        model.model.layers[i].self_attn.scope = scope
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.model.layers[i].self_attn)