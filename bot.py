import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from supabase import create_client

# Force CPU Only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model_id = "Qwen/Qwen1.5-1.8B-Chat"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

def summarize_magma_activity(raw_data):
    # Paksa CPU di dalam fungsi
    device = "cpu"
    model.to(device)
    
    messages = [
        {"role": "system", "content": "Kamu petugas BPBD. Rangkum laporan menjadi satu paragraf."},
        {"role": "user", "content": f"Rangkum: {raw_data}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(cpu)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# Pipeline tetap sama
def process_and_save_data(gunung):
    summary = summarize_magma_activity(f"Status {gunung['level_name']}.")
    print(f"✅ Summary: {summary}")

if __name__ == "__main__":
    g = {"nama": "Gunung Merapi", "key": "merapi", "level_code": 3, "level_name": "Siaga", "cuaca": "Cerah"}
    process_and_save_data(g)
