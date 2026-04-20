import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

# model name
model_name = "distilgpt2"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# set evaluation mode (IMPORTANT)
model.eval()

# move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
fp_32= copy.deepcopy(model)






