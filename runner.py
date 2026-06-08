import os
# FORCE CPU: Kunci sukses kita matikan deteksi GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from supabase import create_client

# 1. SETUP SUPABASE
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. SETUP MODEL LLM (CPU Mode)
model_id = "Qwen/Qwen1.5-1.8B-Chat"
print("Memuat tokenizer dan model ke CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.to("cpu")

# 3. FUNGSI SUMMARIZER
def summarize_magma_activity(raw_data):
    system_prompt = """Kamu adalah petugas BPBD di Indonesia. 
Tugasmu merangkum laporan teknis gunung berapi menjadi SATU PARAGRAF narasi (maksimal 3 kalimat) untuk pengumuman ke warga.
Gunakan Bahasa Indonesia baku. Bahasanya harus menenangkan, mudah dipahami, dan langsung sebutkan status gunung di awal kalimat."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Buat pengumuman kondisi terkini dari data ini: {raw_data}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")
    
    out = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.1,
        repetition_penalty=1.1
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, out)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# 4. PIPELINE FETCH & PUSH KE CLOUD
def process_and_save_data(gunung):
    print(f"\nMemproses {gunung['nama']}...")
    
    # Mock data (bisa diganti data API asli nantinya)
    data_mentah = f"Cuaca {gunung['cuaca']}. Status Level {gunung['level_code']} ({gunung['level_name']}). Asap kawah nihil, gempa vulkanik dalam tercatat normal."
    
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
        print(f"✅ Berhasil! Rangkuman {gunung['nama']} tersimpan di Supabase.")
    except Exception as e:
        print(f"❌ Gagal menyimpan {gunung['nama']} ke database: {e}")

# 5. EKSEKUSI
if __name__ == "__main__":
    daftar_gunung = [
        {"nama": "Gunung Merapi", "key": "merapi", "level_code": 3, "level_name": "Siaga", "cuaca": "Cerah"},
        {"nama": "Gunung Agung", "key": "agung", "level_code": 1, "level_name": "Normal", "cuaca": "Berawan"},
        {"nama": "Gunung Rinjani", "key": "rinjani", "level_code": 2, "level_name": "Waspada", "cuaca": "Cerah"}
    ]
    
    for g in daftar_gunung:
        process_and_save_data(g)
