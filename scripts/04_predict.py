import os
import json
import csv
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import logging

# --- Konfigurasi Awal ---
RETRIEVAL_FILE = Path("data/results/retrieved_cases.json")
CASE_FILE = Path("data/processed/cases.json")
OUTPUT_PREDICTIONS_DIR = Path("data/results/") 
PREDICTION_FILE_NAME_FORMAT = "predictions_{method_name}.csv" # Pola nama file

# Konfigurasi Thresholding (Parameter yang bisa diatur)
# Jika True, prediksi akan berdasarkan threshold. Jika False, akan mengambil top_N_PREDICTED_PASALS.
USE_PREDICTION_THRESHOLD = False
# Ambang batas skor minimal untuk sebuah pasal agar diprediksi (untuk USE_PREDICTION_THRESHOLD = True)
PREDICTION_SCORE_THRESHOLD = 0.5 
# Jika USE_PREDICTION_THRESHOLD = False, ambil N pasal dengan skor tertinggi
TOP_N_PREDICTED_PASALS = 10

# Mengatur logging untuk memberikan informasi, peringatan, dan kesalahan.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Utilitas ---
def initialize_directories(file_path: Path) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{file_path.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {file_path.parent}: {e}")
        return False

def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not file_path.exists():
        logger.error(f"File '{file_path}' not found.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error(f"Invalid data format in '{file_path}'. Expected a JSON array, got {type(data).__name__}.")
            return None
            
        if not data:
            logger.warning(f"No data found in '{file_path}'.")
            return []
            
        logger.info(f"Successfully loaded {len(data)} entries from '{file_path}'.")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from '{file_path}'. Invalid JSON format: {e}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading '{file_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading '{file_path}': {e}")
        return None

def extract_pasals(text: Optional[str]) -> List[str]:
    """
    Mengekstrak semua referensi pasal dari sebuah string teks menggunakan ekspresi reguler.
    Fungsi ini harus konsisten dengan `02_case_representation.py` dan `05_evaluation.py`.
    """
    pasal_pattern = re.compile(
        r"Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?",
        re.IGNORECASE
    )
    pasals = pasal_pattern.findall(text or "")
    pasals_cleaned = []
    for p in pasals:
        p = re.sub(r'\s+', ' ', p).strip()
        p = re.sub(r'Ayat\s*\(\s*(\d+)\s*\)', r'Ayat (\1)', p, re.IGNORECASE)
        p = re.sub(r'huruf\s*([a-zA-Z])', r'huruf \1', p, re.IGNORECASE)
        pasals_cleaned.append(p.title())
    return list(dict.fromkeys(pasals_cleaned))

def weighted_majority_vote(cases_with_scores: List[Tuple[Dict[str, Any], float]]) -> str:
    """
    Melakukan 'majority vote' berbobot untuk menentukan solusi yang diprediksi 
    berdasarkan pasal-pasal dan skor kemiripan dari kasus-kasus yang relevan.
    
    Args:
        cases_with_scores (List[Tuple[Dict[str, Any], float]]): Daftar tuple, 
            di mana setiap tuple berisi (objek kasus, skor kemiripan kasus tersebut).
            
    Returns:
        str: String yang berisi pasal-pasal yang diprediksi, dipisahkan dengan "; ".
             Mengembalikan "N/A" jika tidak ada pasal yang dapat ditemukan.
    """
    pasal_weighted_scores: Dict[str, float] = {}
    
    for case, score in cases_with_scores:
        if not isinstance(case, dict):
            logger.warning(f"Skipping non-dictionary case in weighted majority vote: {case}")
            continue
        
        # Ekstrak pasal dari bidang 'pasal' setiap kasus
        pasals = extract_pasals(case.get("pasal", ""))
        
        # Tambahkan skor kasus ke setiap pasal yang ditemukan
        for pasal in pasals:
            pasal_weighted_scores[pasal] = pasal_weighted_scores.get(pasal, 0.0) + score
    
    # Jika tidak ada pasal yang ditemukan sama sekali dari semua kasus
    if not pasal_weighted_scores:
        return "N/A" 
        
    # Urutkan pasal berdasarkan skor bobotnya secara menurun
    sorted_pasals = sorted(pasal_weighted_scores.items(), key=lambda item: item[1], reverse=True)
    
    predicted_pasals_final = []

    if USE_PREDICTION_THRESHOLD:
        # Ambil pasal yang skor bobotnya melebihi ambang batas
        for pasal, score in sorted_pasals:
            if score >= PREDICTION_SCORE_THRESHOLD:
                predicted_pasals_final.append(pasal)
    else:
        # Ambil TOP_N_PREDICTED_PASALS pasal teratas
        for pasal, score in sorted_pasals[:TOP_N_PREDICTED_PASALS]:
            predicted_pasals_final.append(pasal)
    
    # Jika setelah filter tidak ada pasal yang tersisa, kembalikan "N/A"
    if not predicted_pasals_final:
        return "N/A"

    # Gabungkan pasal-pasal teratas menjadi satu string
    return "; ".join(predicted_pasals_final)

# --- Fungsi Utama ---

def main():
    """
    Fungsi utama yang mengatur alur kerja prediksi untuk setiap metode retrieval.
    """
    if not initialize_directories(OUTPUT_PREDICTIONS_DIR / "dummy.txt"):
        return 

    logger.info("Starting comprehensive prediction process for all retrieval methods...")

    retrieved_data = load_json_data(RETRIEVAL_FILE)
    if retrieved_data is None:
        logger.error("Failed to load retrieval data. Exiting.")
        return
    if not retrieved_data:
        logger.warning("No retrieval data found. Nothing to predict.")
        return

    case_data = load_json_data(CASE_FILE)
    if case_data is None:
        logger.error("Failed to load case base data. Exiting.")
        return
    if not case_data:
        logger.warning("No case base data found. Cannot perform weighted voting.")
        return

    # Buat kamus untuk pencarian kasus yang efisien berdasarkan case_id.
    case_dict = {str(c.get("case_id")): c for c in case_data if c.get("case_id")}
    if not case_dict:
        logger.error("No valid case_ids found in case base. Cannot map retrieved cases.")
        return

    # Kumpulkan semua nama metode retrieval yang ada di dalam retrieved_cases.json
    all_retrieval_methods = set()
    for query_entry in retrieved_data:
        if isinstance(query_entry, dict) and "retrieval_results" in query_entry:
            all_retrieval_methods.update(query_entry["retrieval_results"].keys())
    
    if not all_retrieval_methods:
        logger.error("No retrieval methods found in retrieved_cases.json. Please check its structure.")
        return

    logger.info(f"Found retrieval methods: {', '.join(sorted(list(all_retrieval_methods)))}")

    # Iterasi melalui setiap metode retrieval yang ditemukan
    for method_name in sorted(list(all_retrieval_methods)):
        logger.info(f"\nProcessing predictions for retrieval method: '{method_name}'")
        
        predictions_for_method = []
        processed_queries_method = 0

        for query_entry in retrieved_data:
            if not isinstance(query_entry, dict):
                logger.warning(f"Skipping non-dictionary entry in retrieval data: {query_entry}")
                continue

            query_id = str(query_entry.get("query_id", "UNKNOWN_QUERY"))
            retrieval_results_for_query = query_entry.get("retrieval_results", {})

            # Dapatkan top_k_case_ids dan skor kemiripan spesifik untuk metode ini
            method_data = retrieval_results_for_query.get(method_name, {"case_ids": [], "scores": []})
            top_case_ids_for_method = method_data.get("case_ids", [])
            similarity_scores_for_method = method_data.get("scores", [])

            if not isinstance(top_case_ids_for_method, list) or not isinstance(similarity_scores_for_method, list) or len(top_case_ids_for_method) != len(similarity_scores_for_method):
                logger.warning(f"Skipping query {query_id} for method '{method_name}': Invalid retrieval_results structure. Skipping.")
                continue
            
            cases_for_weighted_vote: List[Tuple[Dict[str, Any], float]] = []
            for i, cid in enumerate(top_case_ids_for_method):
                if cid in case_dict:
                    cases_for_weighted_vote.append((case_dict[cid], similarity_scores_for_method[i]))
                else:
                    logger.warning(f"Case ID '{cid}' for query '{query_id}' and method '{method_name}' not found in case base. Skipping this case for voting.")

            if not cases_for_weighted_vote:
                predicted = "N/A"
                logger.warning(f"No valid top-k cases found for query {query_id} with method '{method_name}'. Setting predicted solution to 'N/A'.")
            else:
                predicted = weighted_majority_vote(cases_for_weighted_vote)
            
            predictions_for_method.append({
                "query_id": query_id,
                "predicted_solution": predicted,
                # Simpan top N ID yang digunakan untuk prediksi dari metode ini (untuk debugging/audit)
                "top_retrieved_case_ids_for_method": top_case_ids_for_method 
            })
            processed_queries_method += 1
        
        logger.info(f"Processed {processed_queries_method} queries for method '{method_name}'.")

        # Simpan hasil prediksi untuk metode ini ke file CSV terpisah
        output_file_path = OUTPUT_PREDICTIONS_DIR / PREDICTION_FILE_NAME_FORMAT.format(method_name=method_name.replace(" ", "_").replace("-", "_"))
        
        try:
            with open(output_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                
                # Header kolom harus disesuaikan dengan field yang disimpan
                writer.writerow(["query_id", "predicted_solution", "top_retrieved_case_ids_for_method"])
                
                for r in predictions_for_method:
                    # Pastikan list top_retrieved_case_ids_for_method dikonversi ke string
                    top_ids_str = ", ".join(map(str, r.get("top_retrieved_case_ids_for_method", [])))
                    writer.writerow([
                        r.get("query_id", ""),
                        r.get("predicted_solution", ""),
                        top_ids_str
                    ])

            logger.info(f"✅ Prediksi untuk metode '{method_name}' berhasil disimpan ke: '{output_file_path}'")
        except Exception as e:
            logger.error(f"❌ Gagal menulis prediksi untuk metode '{method_name}' ke '{output_file_path}': {e}")

# --- Titik Masuk Skrip ---
if __name__ == "__main__":
    main()
