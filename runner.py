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
    # System prompt baru: Objektif, Informatif, Akurat, Tetap 3 Kalimat
    system_prompt = """Kamu adalah sistem pusat informasi bencana. 
Tugasmu merangkum laporan teknis gunung berapi menjadi pengumuman yang informatif, akurat, dan mudah dipahami masyarakat umum.

Patuhi aturan ketat berikut:
1. Sampaikan data secara objektif dan faktual apa adanya, tidak perlu menambahkan kalimat penenang yang dibuat-buat.
2. Hasil rangkuman WAJIB terdiri dari TEPAT 3 KALIMAT (tidak kurang, tidak lebih).
3. Struktur Kalimat:
   - Kalimat 1: Langsung sebutkan nama gunung dan status levelnya saat ini.
   - Kalimat 2: Ringkasan kondisi cuaca dan visual kawah (asap/visual gunung).
   - Kalimat 3: Ringkasan aktivitas kegempaan terkini dari data.
4. Gunakan Bahasa Indonesia baku yang ringkas dan tegas."""

    # Contoh Few-Shot diperbarui agar AI meniru struktur objektif ini
    messages = [
        {"role": "system", "content": system_prompt},
        
        # --- CONTOH FEW-SHOT (Panduan Gaya Bahasa) ---
        {
            "role": "user", 
            "content": "Buat pengumuman kondisi terkini dari data ini: Cuaca Berawan. Status Level II (Waspada). Asap kawah nihil. Terekam 5 kali gempa Vulkanik Dalam dengan amplitudo 10-20 mm."
        },
        {
            "role": "assistant", 
            "content": "Saat ini Gunung berada pada status Level II (Waspada). Berdasarkan pengamatan visual, cuaca di sekitar area gunung terpantau berawan dan asap kawah tidak teramati. Dari data kegempaan, tercatat telah terjadi 5 kali aktivitas gempa vulkanik dalam dengan amplitudo mencapai 10-20 mm."
        },
        # ----------------------------------------------
        
        # Input Data Asli yang Berjalan di Pipeline
        {"role": "user", "content": f"Buat pengumuman kondisi terkini dari data ini: {raw_data}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")
    
    out = model.generate(
        inputs.input_ids,
        max_new_tokens=120, # Ditambah sedikit agar kalimat ke-3 tidak terpotong di tengah jalan
        temperature=0.1,    # Suhu rendah agar AI tetap kaku mengikuti aturan fakta
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
        supabase.table('volcano_summarizer').upsert(payload).execute()
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
