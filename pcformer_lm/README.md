## Our training code is based on huggingface transformers

from configuration_llama import LlamaConfig
from modeling_llama import LlamaForCausalLM
config = LlamaConfig()
model = LlamaForCausalLM(config)

## During the inference phase, you can change the config.json, e.g. autoconfig to load PCformer
"auto_map": {
"AutoConfig": "configuration_llama.LlamaConfig",
"AutoModel": "modeling_llama.LlamaModel",
"AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
}

from transformers import AutoModel
model = AutoModel.from_pretrained("your/saved/model_path")