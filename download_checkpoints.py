import transformers
from transformers import AutoModel
from pathlib import Path

def download_checkpoint(name):
    model = AutoModel.from_pretrained(name, cache_dir=Path("saves", "finetune"))
    output_dir = Path("saves", "finetune", name)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

download_checkpoint("open-unlearning/tofu_Llama-2-7b-chat-hf_retain99")
