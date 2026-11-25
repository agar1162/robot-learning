import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5EncoderModel

MODEL_ROOT = "/scratch/agarcia05/checkpoints/Wan2.1-T2V-1.3B-Diffusers"

def load_text_encoder(device="cuda") -> tuple[AutoTokenizer, UMT5EncoderModel]:          # only for caching instruction embeddings
    tok = AutoTokenizer.from_pretrained(MODEL_ROOT, subfolder="tokenizer")
    t5 = UMT5EncoderModel.from_pretrained(MODEL_ROOT, subfolder="text_encoder",
                                          torch_dtype=torch.bfloat16).to(device).eval()
    return tok, t5

def load_vae(device="cuda") -> AutoencoderKLWan:
    vae = AutoencoderKLWan.from_pretrained(MODEL_ROOT, subfolder="vae",
                                           torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)                  # frozen in Fast-WAM
    return vae

def load_dit(device="cuda") -> WanTransformer3DModel:
    return WanTransformer3DModel.from_pretrained(MODEL_ROOT, subfolder="transformer",
                                                 torch_dtype=torch.bfloat16).to(device)