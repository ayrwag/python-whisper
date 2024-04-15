from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-tiny"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("./openai-whisper-tiny")
processor.save_pretrained("./openai-whisper-tiny")