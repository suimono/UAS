import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from datetime import datetime
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Setup Logging ---
# Mengatur sistem logging untuk mencatat informasi, peringatan, dan kesalahan.
# Log akan ditampilkan di konsol dan juga disimpan ke file log.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Path ---
# Mendefinisikan semua direktori input dan output. Menggunakan pathlib.Path 
# untuk penanganan path yang lebih modern dan platform-agnostik.
INPUT_DIR = Path("data/pdf")
RAW_TXT_DIR = Path("data/raw")

# --- Fungsi Utilitas Direktori ---

def ensure_directories():
    """
    Memastikan semua direktori yang diperlukan untuk input dan output sudah ada.
    Jika sebuah direktori belum ada, fungsi ini akan membuatnya.
    """
    for directory in [INPUT_DIR, RAW_TXT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

# --- Fungsi Generasi Nama File Teks Mentah ---

def generate_raw_text_filename(pdf_filename_stem: str) -> str:
    """
    Menghasilkan nama file teks mentah yang bersih dan unik dari stem nama file PDF.
    Ini akan digunakan sebagai nama file .txt di data/raw/.
    
    Args:
        pdf_filename_stem (str): Nama file PDF tanpa ekstensi (misal: "putusan_123_abc").
        
    Returns:
        str: Nama file yang dibersihkan.
    """
    # Bersihkan nama file: hapus karakter non-alfanumerik atau non-underscore/dash, 
    # ganti spasi dengan underscore, hindari underscore berturut-turut.
    clean_name = re.sub(r'[^\w\-.]', '_', pdf_filename_stem.lower())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name

# --- Fungsi Ekstraksi Teks PDF ---

def extract_text_blocks_improved(page: fitz.Page) -> str:
    """
    Mengekstrak teks dari halaman PDF menggunakan metode 'blocks' PyMuPDF,
    dengan penyortiran yang lebih baik dan filter untuk block yang valid.
    """
    try:
        blocks = page.get_text("blocks")
        if not blocks:
            return ""
        
        # Urutkan blok berdasarkan posisi (dari atas ke bawah, dari kiri ke kanan)
        # Menggunakan pembulatan untuk menangani sedikit perbedaan koordinat.
        blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
        
        text_parts = []
        for block in blocks:
            # Format block: [x0, y0, x1, y1, text, block_no, block_type]
            if len(block) >= 5 and block[4] and isinstance(block[4], str):
                text = block[4].strip()
                if text: # Pastikan teks tidak kosong setelah strip
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.debug(f"Block extraction failed on page (improved): {e}")
        return ""

def extract_text_dict_improved(page: fitz.Page) -> str:
    """
    Mengekstrak teks dari halaman PDF menggunakan metode 'dict' PyMuPDF,
    dengan iterasi yang lebih hati-hati melalui struktur kamus.
    """
    try:
        text_dict = page.get_text("dict")
        if not text_dict or "blocks" not in text_dict:
            return ""
        
        page_text_parts = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            block_lines = []
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                
                line_text = "".join([span.get("text", "") for span in line["spans"]])
                if line_text.strip():
                    block_lines.append(line_text.strip())
            
            if block_lines:
                page_text_parts.append(" ".join(block_lines)) # Gabungkan baris dalam blok dengan spasi
        
        return "\n".join(page_text_parts) # Gabungkan blok dengan newline
    except Exception as e:
        logger.debug(f"Dict extraction failed on page (improved): {e}")
        return ""

def clean_extracted_text(text: str) -> str:
    """
    Membersihkan teks yang diekstrak dari PDF.
    Fokus pada normalisasi spasi/baris baru dan penghapusan artefak yang sangat jelas.
    Tujuan utama adalah mempertahankan sebanyak mungkin teks asli tanpa pemotongan yang tidak disengaja.
    
    Args:
        text (str): Teks mentah yang diekstrak dari PDF.
        
    Returns:
        str: Teks yang sudah bersih dan dinormalisasi dengan filtering minimal.
    """
    if not text:
        return ""
    
    # 1. Normalisasi spasi dan baris baru
    text = text.replace('\x00', ' ').replace('\xa0', ' ') # Hapus karakter null dan non-breaking space
    text = re.sub(r'[\r\n]+', '\n', text) # Normalisasi CRLF ke LF, hapus multiple newlines menjadi single newline
    text = re.sub(r'[ \t]+', ' ', text) # Normalisasi spasi dan tab ganda menjadi single space
    text = text.strip() # Hapus spasi/newline di awal/akhir setelah normalisasi

    if not text: # Jika teks menjadi kosong setelah normalisasi awal
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line: # Lewati baris yang kosong setelah di-strip
            continue
        
        # 2. Filter baris yang tidak diinginkan (artefak yang sangat spesifik dan jelas)
        # Filter ini dirancang untuk sangat konservatif, hanya membuang yang pasti artefak.
        
        # Filter nomor halaman yang sangat jelas:
        # Contoh: "1", "- 2 -", "-- 3 --", "Page 4"
        if re.fullmatch(r'^\s*[-_]?\s*\d+\s*[-_]?\s*$', line, re.IGNORECASE) or \
           re.fullmatch(r'^\s*[Pp][Aa][Gg][Ee]\s+\d+\s*$', line, re.IGNORECASE):
            continue
        
        # Filter garis horizontal atau deretan simbol sederhana yang berulang:
        # Contoh: "------------", "*********", "========"
        # Hanya filter jika baris *seluruhnya* terdiri dari satu atau dua karakter non-alphanumeric berulang.
        if len(line) > 5: # Harus cukup panjang agar dianggap garis pemisah
            if all(not char.isalnum() and not char.isspace() for char in line):
                if len(set(char for char in line if not char.isspace())) <= 2:
                    continue

        # Filter baris yang sangat pendek dan tidak mengandung huruf/angka (seringkali sisa-sisa simbol atau numbering yang salah)
        if len(line) <= 3 and not re.search(r'[a-zA-Z0-9]', line):
            continue
        
        # Filter Roman numerals jika tampak sebagai penomoran list/sub-bagian yang berdiri sendiri.
        # Contoh: "I.", "II.", "V."
        if len(line) <= 5 and re.fullmatch(r'^[ivxlc]+\.$', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    # Gabungkan baris yang bersih dengan satu newline.
    return "\n".join(cleaned_lines).strip()


def extract_text_from_pdf(file_path: Path) -> Optional[str]:
    """
    Mengekstrak teks dari file PDF menggunakan berbagai metode PyMuPDF 
    dan memilih hasil terbaik setelah pembersihan.
    
    Args:
        file_path (Path): Path ke file PDF.
        
    Returns:
        Optional[str]: Seluruh teks yang diekstrak dari PDF, atau None jika gagal.
    """
    try:
        doc = fitz.open(file_path)
        full_text_parts = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text_candidates = {}
                
                # Coba berbagai metode ekstraksi dan simpan hasilnya
                methods = {
                    "standard_sorted": lambda p: p.get_text("text", sort=True),
                    "standard_unsorted": lambda p: p.get_text("text"),
                    "blocks_custom": extract_text_blocks_improved,
                    "dict_custom": extract_text_dict_improved
                }
                
                for method_name, method_func in methods.items():
                    try:
                        result = method_func(page)
                        if result and result.strip():
                            page_text_candidates[method_name] = result
                    except Exception as e:
                        logger.debug(f"Method '{method_name}' failed on page {page_num + 1} of {file_path.name}: {e}")
                
                best_page_text = ""
                # Pilih teks terbaik: yang paling panjang (setelah dibersihkan)
                for method_name, text in page_text_candidates.items():
                    cleaned_current_text = clean_extracted_text(text)
                    if len(cleaned_current_text) > len(best_page_text):
                        best_page_text = cleaned_current_text
                        logger.debug(f"Page {page_num + 1}: Using {method_name} method (length {len(best_page_text)})")
                
                if best_page_text:
                    full_text_parts.append(f"\n=== HALAMAN {page_num + 1} ===\n") # Penanda halaman
                    full_text_parts.append(best_page_text)
                else:
                    logger.warning(f"No meaningful text extracted from page {page_num + 1} of {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} of {file_path.name}: {e}", exc_info=True)
                continue 
        
        doc.close()
        return "\n".join(full_text_parts).strip()
        
    except fitz.FileDataError as e:
        logger.error(f"PDF file corrupted or invalid: {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error opening or processing PDF {file_path}: {e}", exc_info=True)
        return None

# --- Fungsi Penyimpanan File ---

def save_text_file(content: str, file_path: Path) -> bool:
    """
    Menyimpan konten teks ke file yang ditentukan.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving text file {file_path}: {e}")
        return False

# --- Fungsi Utama Pemrosesan ---

def main():
    """
    Fungsi utama untuk mengekstrak teks dari semua file PDF di INPUT_DIR,
    membersihkannya, dan menyimpan ke file teks mentah di data/raw/.
    """
    try:
        # Langkah 1: Pastikan semua direktori yang diperlukan ada.
        ensure_directories()
        
        # Langkah 2: Periksa apakah folder input PDF ada dan berisi file.
        if not INPUT_DIR.exists():
            logger.error(f"Input directory not found: {INPUT_DIR}")
            print(f"Silakan buat folder '{INPUT_DIR}' dan masukkan file PDF di dalamnya.")
            return
        
        # Langkah 3: Dapatkan daftar semua file PDF di direktori input.
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {INPUT_DIR}")
            print(f"Tidak ada file PDF di folder '{INPUT_DIR}'")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process.")
        print(f"� Ditemukan {len(pdf_files)} file PDF untuk diproses.")
        
        # Mengumpulkan daftar file teks mentah yang sudah ada untuk menghindari pemrosesan ulang.
        existing_raw_txt_files = {f.stem for f in RAW_TXT_DIR.glob("*.txt")}
        logger.info(f"Found {len(existing_raw_txt_files)} existing raw text files in {RAW_TXT_DIR}.")

        # Variabel untuk melacak status pemrosesan
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Langkah 4: Iterasi (ulang) melalui setiap file PDF.
        for pdf_file_path in tqdm(pdf_files, desc="Memproses PDF"):
            try:
                filename = pdf_file_path.name
                file_stem = generate_raw_text_filename(pdf_file_path.stem) # Nama file .txt yang akan disimpan
                
                # Cek apakah file ini sudah diproses ke raw .txt sebelumnya
                if file_stem in existing_raw_txt_files:
                    logger.info(f"File {filename} (raw text stem: {file_stem}) already has a corresponding raw text file, skipping.")
                    skipped_count += 1
                    continue
                
                logger.info(f"Processing: {filename}")
                
                # Ekstrak teks dari PDF
                full_text = extract_text_from_pdf(pdf_file_path)
                
                # Periksa apakah ekstraksi berhasil dan teks cukup panjang.
                if not full_text or len(full_text.strip()) < 200: # Batas minimal 200 karakter
                    logger.error(f"Failed to extract sufficient text from {filename} or text too short.")
                    failed_count += 1
                    continue
                
                # Simpan teks mentah ke file .txt di direktori RAW_TXT_DIR
                txt_filename_path = RAW_TXT_DIR / f"{file_stem}.txt"
                if not save_text_file(full_text, txt_filename_path):
                    failed_count += 1
                    continue # Lanjutkan ke file berikutnya jika gagal menyimpan
                
                processed_count += 1
                logger.info(f"Successfully processed {filename} -> {txt_filename_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True) # exc_info=True untuk traceback
                failed_count += 1
                continue
        
        # Langkah 5: Tampilkan ringkasan pemrosesan.
        print(f"\n=== RINGKASAN PEMROSESAN ===")
        print(f"Total file PDF ditemukan: {len(pdf_files)}")
        print(f"Berhasil dikonversi ke teks: {processed_count}")
        print(f"Gagal dikonversi: {failed_count}")
        print(f"Dilewati (sudah ada raw text-nya): {skipped_count}")
        print(f"File teks mentah disimpan di: {RAW_TXT_DIR}")
        print("=============================")
            
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}", exc_info=True)
        print(f"Terjadi error kritis pada pemrosesan: {e}")

# --- Titik Masuk Skrip ---
# Memastikan fungsi main() dijalankan hanya ketika skrip dieksekusi langsung.
if __name__ == "__main__":
    main()
