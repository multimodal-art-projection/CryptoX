# python自带的库
from collections import defaultdict, OrderedDict
# 常用的开源库
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
from safetensors.torch import load_file

def get_device_with_most_free_memory():
    """
    获取拥有最多空闲显存的CUDA设备,如果没有可用的GPU，则返回CPU
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    devices = list(range(torch.cuda.device_count()))
    device_memory = []
    
    for device in devices:
        torch.cuda.set_device(device)
        # 获取当前设备的总显存
        total_mem = torch.cuda.get_device_properties(device).total_memory
        # 获取当前设备已分配的显存
        allocated_mem = torch.cuda.memory_allocated(device)
        # 计算空闲显存
        free_mem = total_mem - allocated_mem
        device_memory.append((device, free_mem))
    
    # 按空闲显存从多到少排序
    device_memory.sort(key=lambda x: x[1], reverse=True)
    # 返回空闲显存最多的设备
    return torch.device(f'cuda:{device_memory[0][0]}')

def load_tokenizer_only(model_size, hf_token=None):
    """
    读取toenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(f'/dlabdata1/llama2_hf/Llama-2-{model_size}-hf', use_auth_token=hf_token)
    return tokenizer


class AttnWrapper(torch.nn.Module):
    """
    包装注意力层（Attn）以进行干预和激活管理
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False
    def forward(self, *args, **kwargs):
        """
        前向传播，修改注意力的行为以实现干预
        """
        if self.act_as_identity:
            kwargs['attention_mask'] += kwargs['attention_mask'][0, 0, 0, 1]*torch.tril(torch.ones(kwargs['attention_mask'].shape,
                                                                                                   dtype=kwargs['attention_mask'].dtype,
                                                                                                   device=kwargs['attention_mask'].device),
                                                                                        diagonal=-1)
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        """
        重置干预设置
        """
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False

class MLPWrapper(torch.nn.Module):
    """
    包装MLP层，以便进行神经元干预和激活管理
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.up_proj = mlp.up_proj
        self.gate_proj = mlp.gate_proj
        self.act_fn = mlp.act_fn
        self.down_proj = mlp.down_proj
        self.neuron_interventions = {}
        self.post_activation = None
    
    def forward(self, x):
        """
        前向传播，通过MLP计算输出，并应用干预
        """
        post_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        self.post_activation = post_activation.detach().cpu()
        output = self.down_proj(post_activation)
        if len(self.neuron_interventions) > 0:
            print('performing intervention: mlp_neuron_interventions')
            for neuron_idx, mean in self.neuron_interventions.items():
                output[:, :, neuron_idx] = mean
        return output
    
    def reset(self):
        """
        重置神经元干预设置
        """
        self.neuron_interventions = {}
    
    def freeze_neuron(self, neuron_idx, mean):
        """
        冻结指定的神经元，并将其输出设置为给定的均值
        """
        self.neuron_interventions[neuron_idx] = mean

class BlockOutputWrapper(torch.nn.Module):
    """
    包装Transformer层的输出，以便进行干预并管理各层的激活和输出
    """
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_post_activation = None 
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        self.add_to_last_tensor = None
        self.output = None
        self.output_normalized = None

    def forward(self, *args, **kwargs):
        """
        前向传播，计算每个子层的输出并管理干预
        """
        free_device = get_device_with_most_free_memory()
        output = self.block(*args, **kwargs)
        if self.add_to_last_tensor is not None:
            print('performing intervention: add_to_last_tensor')
            output[0][:, -1, :] += self.add_to_last_tensor
        self.output = output[0]
        self.output_normalized = self.norm(output[0].to(self.norm.weight.device))
        self.block_output_unembedded = self.unembed_matrix(self.output_normalized.to(self.unembed_matrix.weight.device))
        self.output_normalized.to(free_device)
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output.to(self.unembed_matrix.weight.device)))
        attn_output += args[0].to(attn_output.device)
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output.to(self.post_attention_layernorm.weight.device)))
        attn_output.to(free_device)
        self.mlp_post_activation = self.block.mlp.post_activation
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output.to(self.unembed_matrix.weight.device)))
        self.mlp_output_unembedded.to(free_device)
        return output

    def mlp_freeze_neuron(self, neuron_idx, mean):
        """
        冻结MLP层中指定神经元的输出
        """
        self.block.mlp.freeze_neuron(neuron_idx, mean) 

    def block_add_to_last_tensor(self, tensor):
        """
        向最后一个token的输出中添加干预张量
        """
        self.add_to_last_tensor = tensor

    def attn_add_tensor(self, tensor):
        """
        向注意力输出中添加干预张量
        """
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        """
        重置所有干预设置
        """
        self.block.self_attn.reset()
        self.block.mlp.reset()
        self.add_to_last_tensor = None

    def get_attn_activations(self):
        """
        获取当前层的注意力激活值
        """
        return self.block.self_attn.activations

class LlamaHelper:
    """
    提供了多种操作模型的功能，如生成文本、干预神经元等
    """
    def __init__(self, dir='/dlabdata1/llama2_hf/Llama-2-7b-hf', hf_token=None, device=None, load_in_4bit=True, use_embed_head=False, device_map='auto'):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(dir, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(dir, use_auth_token=hf_token,
                                                                device_map=device_map,
                                                                quantization_config=quantization_config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(dir, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(dir, use_auth_token=hf_token,
                                                                device_map=device_map,
                                                                torch_dtype='auto')
        self.use_embed_head = True
        W = list(self.model.model.embed_tokens.parameters())[0].detach()
        self.head_embed = torch.nn.Linear(W.size(1), W.size(0), bias=False)
        self.head_embed.to(W.dtype)
        self.norm = self.model.model.norm
        with torch.no_grad():
            self.head_embed.weight.copy_(W) 
        self.head_embed.to(self.model.device)
        self.head_unembed = self.model.lm_head
        self.device = next(self.model.parameters()).device
        if use_embed_head:
            head = self.head_embed
        else:
            head = self.head_unembed
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, head, self.model.model.norm)


    def set_embed_head(self):
        """
        设置embed head
        """
        self.use_embed_head = True
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].unembed_matrix = self.head_embed

    def set_unembed_head(self):
        """
        设置unembed_head
        """
        self.use_embed_head = False
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].unembed_matrix = self.head_unembed
    
    def generate_text(self, prompt, max_length=100):
        """
        根据给定的提示（prompt）生成文本，最大长度为 max_length
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    def generate_intermediate_text(self, layer_idx, prompt, max_length=100, temperature=0.0):
        """
        生成中间层文本
        """
        layer = self.model.model.layers[layer_idx]
        for _ in range(max_length):
            self.get_logits(prompt)
            next_id = self.sample_next_token(layer.block_output_unembedded[:,-1], temperature=temperature)
            prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[1:]+[next_id])
            if next_id == self.tokenizer.eos_token_id:
                break
        return prompt

    def sample_next_token(self, logits, temperature=1.0):
        """
        采样下一个 token，使用温度调节采样策略
        """
        assert temperature >= 0, "temp must be geq 0"
        if temperature == 0:
            return self._sample_greedy(logits)
        return self._sample_basic(logits/temperature)
        
    def _sample_greedy(self, logits):
        """
        贪心采样策略，选择最大值作为下一个 token
        """
        return logits.argmax().item()

    def _sample_basic(self, logits):
        """
        基础采样策略，根据概率分布进行采样
        """
        return torch.distributions.categorical.Categorical(logits=logits).sample().item()
    
    def get_logits(self, prompt):
        """
        获取给定提示的 logits（输出的概率分布）
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
          logits = self.model(inputs.input_ids).logits
          return logits

    def set_neuron_intervention(self, layer_idx, neuron_idx, mean):
        """
        对特定层的神经元进行干预，冻结该神经元并设置均值
        """
        self.model.model.layers[layer_idx].mlp_freeze_neuron(neuron_idx, mean)

    def set_add_attn_output(self, layer, add_output):
        """
        为特定层的注意力机制添加额外的输出
        """
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        """
        获取特定层的注意力激活（attention activations）
        """
        return self.model.model.layers[layer].get_attn_activations()

    def set_add_to_last_tensor(self, layer, tensor):
        """
        为特定层的最后一个 token 添加额外的张量
        """
        print('setting up intervention: add tensor to last soft token')
        self.model.model.layers[layer].block_add_to_last_tensor(tensor)

    def reset_all(self):
        """
        重置所有模型层的状态
        """
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        """
        打印解码后的激活值
        """
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(indices.detach().cpu().numpy().tolist(), tokens, probs_percent)))

    def logits_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_block=True):
        """
        获取所有层的 logits 并返回相关信息
        """
        res = defaultdict(list)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            if return_block:
                res['block_2'] += [layer.block_output_unembedded.detach().cpu()]
            if return_attn_mech:
                res['attn'] += [layer.attn_mech_output_unembedded.detach().cpu()]
            if return_intermediate_res:
                res['block_1'] += [layer.intermediate_res_unembedded.detach().cpu()]
            if return_mlp:
                res['mlp'] += [layer.mlp_output_unembedded.detach().cpu()]
        for k,v in res.items():
            res[k] = torch.cat(v, dim=0)
        if len(res) == 1:
            return list(res.values())[0]
        return res

    def latents_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_mlp_post_activation=False, return_block=True, normalized=False):
        """
        获取所有层的潜在表示（latent representations）
        """
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.get_logits(text)
        tensors = []
        if return_block:
            for i, layer in enumerate(self.model.model.layers):
                if normalized:
                    tensors += [layer.output.detach().cpu()]
                else:
                    tensors += [layer.output_normalized.detach().cpu()]
        elif return_mlp_post_activation:
            for i, layer in enumerate(self.model.model.layers):
                tensors += [layer.mlp_post_activation.detach().cpu()]
        return torch.cat(tensors, dim=0)
        
    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        """
        解码所有层的输出并打印
        """
        print('Prompt:', text)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output')