import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from supabase import create_client

# ... (lanjutan kode lainnya)

# 1. SETUP KREDENSIAL (Diambil dari GitHub Secrets atau Environment Variable)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Error: SUPABASE_URL atau SUPABASE_KEY tidak ditemukan di environment!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. SETUP MODEL LLM (Tanpa 4-bit, langsung loading ke CPU)
model_id = "Qwen/Qwen1.5-1.8B-Chat"

print("Memuat tokenizer dan model ke CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load langsung ke CPU (tanpa bnb_config)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", # Membiarkan torch memilih tipe data yang pas
    device_map="cpu",   # Paksa jalan di CPU
    trust_remote_code=True
)

# 3. FUNGSI SUMMARIZER
def summarize_magma_activity(raw_data):
    system_prompt = """Kamu adalah petugas BPBD di Indonesia. 
Tugasmu merangkum laporan teknis gunung berapi menjadi SATU PARAGRAF narasi (maksimal 3 kalimat) untuk pengumuman ke warga.
Gunakan Bahasa Indonesia baku (bukan Bahasa Melayu). 
Bahasanya harus menenangkan, mudah dipahami, dan langsung sebutkan status gunung di awal kalimat."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Buat pengumuman kondisi terkini dari data ini: Gunung jelas. Asap kawah nihil. Cuaca berawan. Terekam 5 kali gempa Vulkanik Dalam dengan amplitudo 10-20 mm, durasi 15 detik. Status Level II (Waspada)."},
        {"role": "assistant", "content": "Saat ini Gunung berada pada status Waspada (Level II). Berdasarkan pantauan terkini, cuaca berawan dan masih terekam beberapa kali aktivitas gempa vulkanik di dalam gunung. Warga diimbau untuk tetap tenang dan mematuhi batas jarak aman dari kawah."},
        {"role": "user", "content": f"Buat pengumuman kondisi terkini dari data ini: {raw_data}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100, 
        temperature=0.1,    
        repetition_penalty=1.1
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# 4. PIPELINE UTAMA
def process_and_save_data(gunung):
    print(f"\nMemproses {gunung['nama']}...")
    
    # Mock data (nanti bisa diganti dengan request real ke API MAGMA)
    data_mentah = f"Cuaca {gunung['cuaca']}. Status Level {gunung['level_code']} ({gunung['level_name']}). Aktivitas terpantau normal."
    
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
        print(f"✅ Berhasil menyimpan {gunung['nama']} ke Cloud.")
    except Exception as e:
        print(f"❌ Gagal menyimpan {gunung['nama']}: {e}")

# 5. EKSEKUSI
daftar_gunung = [
    {"nama": "Gunung Merapi", "key": "merapi", "level_code": 3, "level_name": "Siaga", "cuaca": "Cerah"},
    {"nama": "Gunung Agung", "key": "agung", "level_code": 1, "level_name": "Normal", "cuaca": "Berawan"},
    {"nama": "Gunung Rinjani", "key": "rinjani", "level_code": 2, "level_name": "Waspada", "cuaca": "Cerah"}
]

if __name__ == "__main__":
    for g in daftar_gunung:
        process_and_save_data(g)
