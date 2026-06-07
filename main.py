import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from supabase import create_client

# Setup Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Setup Model - Paksa CPU
model_id = "Qwen/Qwen1.5-1.8B-Chat"
print("Loading model ke CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# PENTING: Hapus device_map="auto" atau "cpu" di sini agar tidak memicu library accelerate
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32, 
    trust_remote_code=True
)

def summarize_magma_activity(raw_data):
    # System prompt
    system_prompt = "Kamu petugas BPBD. Rangkum laporan gunung berapi menjadi satu paragraf."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Rangkum: {raw_data}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Deteksi device secara manual
    device = "cpu"
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model.to(device) # Paksa model ke CPU
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# Pipeline
def process_and_save_data(gunung):
    data_mentah = f"Status {gunung['level_name']}. Aktivitas normal."
    summary = summarize_magma_activity(data_mentah)
    
    payload = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "volcano_name": gunung["nama"],
        "volcano_key": gunung["key"],
        "level_code": gunung["level_code"],
        "level_name": gunung["level_name"],
        "period_start": "00:00:00",
        "period_end": "24:00:00",
        "summary": summary,
        "weather": gunung["cuaca"]
    }
    
    try:
        supabase.table('volcano_summarizer').insert(payload).execute()
        print(f"✅ Sukses: {gunung['nama']}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Eksekusi
if __name__ == "__main__":
    g = {"nama": "Gunung Merapi", "key": "merapi", "level_code": 3, "level_name": "Siaga", "cuaca": "Cerah"}
    process_and_save_data(g)
