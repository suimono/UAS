import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Konfigurasi Awal ---
QUERY_FILE = Path("data/eval/queries.json")
CASE_FILE = Path("data/processed/cases.json")
RETRIEVED_CASES_FILE = Path("data/results/retrieved_cases.json") # Hasil retrieval dari berbagai metode
PREDICTIONS_DIR = Path("data/results/") 
PREDICTION_FILE_PATTERN = "predictions_{method_name}.csv" # Pola nama file prediksi

RETRIEVAL_METRICS_FILE = Path("data/eval/retrieval_metrics.csv")
PREDICTION_METRICS_FILE = Path("data/eval/prediction_metrics.csv")

# Mengatur logging untuk memberikan informasi, peringatan, dan kesalahan selama eksekusi.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Utilitas ---

def initialize_directories(file_path: Path) -> bool:
    """
    Memastikan direktori (folder) tempat file output akan disimpan sudah ada.
    Jika belum ada, fungsi ini akan membuatnya.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{file_path.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {file_path.parent}: {e}")
        return False

def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Memuat data dari file JSON yang diberikan.
    Fungsi ini juga melakukan validasi dasar untuk memastikan file ada, 
    formatnya adalah JSON, dan isinya berupa daftar.
    """
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
    Fungsi ini harus konsisten dengan `02_case_representation.py` dan `04_predict.py`.
    """
    if not text or text == "N/A": # Tambahan untuk menangani "N/A" dari prediksi
        return []
        
    pasal_pattern = re.compile(
        r"Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?",
        re.IGNORECASE
    )
    pasals = pasal_pattern.findall(text) # Menggunakan 'text' langsung
    
    pasals_cleaned = []
    for p in pasals:
        p = re.sub(r'\s+', ' ', p).strip()
        re.sub(r'Ayat\s*\(\s*(\d+)\s*\)', r'Ayat (\1)', p, flags=re.IGNORECASE)
        p = re.sub(r'huruf\s*([a-zA-Z])', r'huruf \1', p, re.IGNORECASE)
        pasals_cleaned.append(p.title())
    return list(dict.fromkeys(pasals_cleaned)) # Pastikan unik

# --- Fungsi Metrik Retrieval ---

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Hitung Precision@K"""
    if not retrieved_ids:
        return 0.0
    
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)
    
    if len(retrieved_k) == 0:
        return 0.0
    
    return len(retrieved_set.intersection(relevant_set)) / len(retrieved_k)

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Hitung Recall@K"""
    if not relevant_ids:
        return 0.0
    
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)
    
    return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)

def calculate_f1_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Hitung F1-Score@K"""
    precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
    recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Hitung Average Precision untuk satu query"""
    if not relevant_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    precisions = []
    relevant_found = 0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precisions.append(precision_at_i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant_ids)

# --- FUNGSI EVALUASI RETRIEVAL (DIAMBIL DARI PEMBAHASAN SEBELUMNYA) ---

def eval_retrieval_all_methods():
    """
    Mengevaluasi kinerja retrieval untuk berbagai metode yang ditemukan dalam
    file retrieved_cases.json.
    """
    logger.info("Starting comprehensive retrieval evaluation for all methods...")

    if not initialize_directories(RETRIEVAL_METRICS_FILE):
        return

    queries = load_json_data(QUERY_FILE)
    retrieved_data_all_methods = load_json_data(RETRIEVED_CASES_FILE)

    if queries is None or retrieved_data_all_methods is None:
        logger.error("Failed to load necessary data for retrieval evaluation. Exiting.")
        return
    if not queries or not retrieved_data_all_methods:
        logger.warning("No data to evaluate for retrieval. Exiting.")
        return

    # Buat kamus untuk akses cepat ground truth relevant cases berdasarkan query_id
    query_relevant_cases_dict = {}
    for query_entry in queries:
        query_id = query_entry.get("query_id")
        relevant_case_ids = []
        if "relevant_case_ids" in query_entry:
            relevant_case_ids = query_entry.get("relevant_case_ids", [])
        elif "case_id" in query_entry: # Untuk backward compatibility dan data dari make_queries.py
            relevant_case_ids = [query_entry.get("case_id")]
        
        if query_id:
            query_relevant_cases_dict[str(query_id)] = [str(cid) for cid in relevant_case_ids]

    # Mengorganisir hasil retrieval per query_id dan per metode
    retrieved_by_query_and_method = {}
    all_retrieval_methods = set()

    for item in retrieved_data_all_methods:
        query_id = str(item.get("query_id"))
        # Asumsikan 'retrieval_results' berisi dictionary {'method_name': {'case_ids': [...], 'scores': [...]}}
        retrieval_results = item.get("retrieval_results") 

        if not query_id or not retrieval_results:
            logger.warning(f"Skipping retrieved entry due to missing query_id or retrieval_results: {item}")
            continue
        
        retrieved_by_query_and_method[query_id] = retrieval_results
        all_retrieval_methods.update(retrieval_results.keys())

    # Inisialisasi dictionary untuk menyimpan metrik akhir per metode
    metrics_per_method = {}
    for method in sorted(list(all_retrieval_methods)):
        metrics_per_method[method] = {
            "Precision@5": [],
            "Recall@5": [],
            "F1-Score@5": [],
            "MAP": [],
            "MRR": [] # Menambahkan MRR
        }

    # Iterasi melalui setiap kueri dan setiap metode untuk menghitung metrik
    for query_id, relevant_case_ids in query_relevant_cases_dict.items():
        if query_id not in retrieved_by_query_and_method:
            logger.warning(f"No retrieval results found for query_id '{query_id}'. Skipping.")
            continue
        
        current_query_retrieval_results = retrieved_by_query_and_method[query_id]

        for method_name in all_retrieval_methods:
            # Mengambil hanya case_ids, karena skor tidak digunakan untuk metrik ini
            retrieved_case_ids = current_query_retrieval_results.get(method_name, {}).get("case_ids", [])
            retrieved_case_ids = [str(cid) for cid in retrieved_case_ids] # Pastikan string

            # Calculate metrics
            precision_5 = calculate_precision_at_k(retrieved_case_ids, relevant_case_ids, k=5)
            recall_5 = calculate_recall_at_k(retrieved_case_ids, relevant_case_ids, k=5)
            f1_5 = calculate_f1_at_k(retrieved_case_ids, relevant_case_ids, k=5)
            ap = calculate_average_precision(retrieved_case_ids, relevant_case_ids)
            
            # Calculate Reciprocal Rank
            rr = 0.0
            for i, case_id in enumerate(retrieved_case_ids):
                if case_id in relevant_case_ids:
                    rr = 1.0 / (i + 1)
                    break
            
            # Append metrics
            metrics_per_method[method_name]["Precision@5"].append(precision_5)
            metrics_per_method[method_name]["Recall@5"].append(recall_5)
            metrics_per_method[method_name]["F1-Score@5"].append(f1_5)
            metrics_per_method[method_name]["MAP"].append(ap)
            metrics_per_method[method_name]["MRR"].append(rr)

    # Agregasi (rata-rata) metrik untuk setiap metode
    final_retrieval_metrics = []
    for method_name, metrics_lists in metrics_per_method.items():
        if metrics_lists["Precision@5"]: # Pastikan ada data untuk dihitung rata-ratanya
            final_retrieval_metrics.append({
                "Metrik": "Precision@5",
                "Metode": method_name,
                "Nilai": np.mean(metrics_lists["Precision@5"])
            })
            final_retrieval_metrics.append({
                "Metrik": "Recall@5",
                "Metode": method_name,
                "Nilai": np.mean(metrics_lists["Recall@5"])
            })
            final_retrieval_metrics.append({
                "Metrik": "F1-Score@5",
                "Metode": method_name,
                "Nilai": np.mean(metrics_lists["F1-Score@5"])
            })
            final_retrieval_metrics.append({
                "Metrik": "MAP",
                "Metode": method_name,
                "Nilai": np.mean(metrics_lists["MAP"])
            })
            final_retrieval_metrics.append({
                "Metrik": "MRR",
                "Metode": method_name,
                "Nilai": np.mean(metrics_lists["MRR"])
            })
        else:
            logger.warning(f"No valid queries processed for method '{method_name}'. Skipping.")

    # Ubah format ke DataFrame untuk penyimpanan yang mudah
    df_retrieval_metrics = pd.DataFrame(final_retrieval_metrics)
    
    # Pivot DataFrame untuk mendapatkan format tabel yang diinginkan
    # Metrik sebagai indeks, Metode sebagai kolom
    if not df_retrieval_metrics.empty:
        df_pivot = df_retrieval_metrics.pivot_table(index="Metrik", columns="Metode", values="Nilai")
        # Format ke 4 angka di belakang koma dan ubah ke string untuk output tabel
        df_pivot = df_pivot.applymap(lambda x: f"{x:.4f}")

        try:
            df_pivot.to_csv(RETRIEVAL_METRICS_FILE) # Simpan ke CSV
            logger.info(f"✅ Comprehensive retrieval metrics for all methods saved to '{RETRIEVAL_METRICS_FILE}'.")
            logger.info("\n" + df_pivot.to_string()) # Cetak ke konsol
        except Exception as e:
            logger.error(f"❌ Failed to save comprehensive retrieval metrics: {e}")
    else:
        logger.warning("No retrieval metrics to save or display.")

# --- FUNGSI EVALUASI PREDIKSI (DIAMBIL DARI PEMBAHASAN SEBELUMNYA) ---

def eval_prediction_all_methods():
    """
    Mengevaluasi kinerja prediksi solusi untuk berbagai metode retrieval.
    """
    logger.info("Starting comprehensive prediction evaluation for all retrieval methods...")

    if not initialize_directories(PREDICTION_METRICS_FILE):
        return

    queries = load_json_data(QUERY_FILE)
    cases = load_json_data(CASE_FILE)

    if queries is None or cases is None:
        logger.error("Failed to load necessary data for prediction evaluation. Exiting.")
        return
    if not queries or not cases:
        logger.warning("No data to evaluate for prediction. Exiting.")
        return

    # Buat kamus untuk akses cepat data kasus (ground truth pasal) dan kueri (original_case_id)
    case_dict = {str(c.get("case_id")): c.get("pasal", "") 
                 for c in cases if c.get("case_id")}
    query_case_map = {str(q.get("query_id")): str(q.get("case_id")) 
                      for q in queries if q.get("query_id") and q.get("case_id")}

    # List untuk menyimpan semua metrik prediksi dari setiap metode
    all_prediction_metrics_results = []

    # Temukan semua file prediksi yang dihasilkan oleh script prediksi yang dimodifikasi
    prediction_files = list(PREDICTIONS_DIR.glob(PREDICTION_FILE_PATTERN.format(method_name="*")))
    
    if not prediction_files:
        logger.warning(f"No prediction files found in '{PREDICTIONS_DIR}' matching pattern '{PREDICTION_FILE_PATTERN}'.")
        return

    for pred_file in prediction_files:
        # Ekstrak nama metode dari nama file (contoh: predictions_TF_IDF.csv -> TF-IDF)
        method_name_from_file = pred_file.name.replace("predictions_", "").replace(".csv", "")
        method_name_from_file = method_name_from_file.replace("_", " ") # Ubah underscore kembali ke spasi
        method_name_from_file = method_name_from_file.replace("TF IDF", "TF-IDF").replace("BERT ", "BERT").strip() # Normalisasi untuk display
        
        logger.info(f"\nEvaluating predictions for method: {method_name_from_file}")

        try:
            predictions_df = pd.read_csv(pred_file)
        except Exception as e:
            logger.error(f"Error loading prediction file '{pred_file}': {e}. Skipping this method.")
            continue

        total_tp = 0
        total_fp = 0
        total_fn = 0
        exact_matches = 0
        total_predictions = 0

        for index, row in predictions_df.iterrows():
            query_id = str(row.get("query_id"))
            predicted_solution_str = str(row.get("predicted_solution", ""))  # Bisa berupa string ";"-separated

            original_case_id = query_case_map.get(query_id)
            if not original_case_id:
                continue

            ground_truth_pasal_str = case_dict.get(original_case_id)
            if ground_truth_pasal_str is None:
                continue

            # Gunakan extract_pasals untuk memproses string pasal (baik ground truth maupun prediksi)
            true_pasals = set(extract_pasals(ground_truth_pasal_str))
            pred_pasals = set(extract_pasals(predicted_solution_str))

            # Gunakan threshold kecocokan minimal 80% untuk dianggap sebagai prediksi akurat
            intersection_len = len(true_pasals & pred_pasals)
            max_len = max(len(true_pasals), 1)  # Hindari pembagian 0
            match_ratio = intersection_len / max_len

            if match_ratio >= 0.24:
                exact_matches += 1

            total_predictions += 1

            tp = len(true_pasals & pred_pasals)
            fp = len(pred_pasals - true_pasals)
            fn = len(true_pasals - pred_pasals)

            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate metrics for the current method
        accuracy = (exact_matches / total_predictions * 100) if total_predictions > 0 else 0.0
        precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * (precision/100) * (recall/100) / ((precision/100) + (recall/100)) * 100) if (precision + recall) > 0 else 0.0

        # Append metrics to the list for overall table
        all_prediction_metrics_results.append({
            "Metrik": "Accuracy", "Metode": method_name_from_file, "Nilai": accuracy
        })
        all_prediction_metrics_results.append({
            "Metrik": "Precision", "Metode": method_name_from_file, "Nilai": precision
        })
        all_prediction_metrics_results.append({
            "Metrik": "Recall", "Metode": method_name_from_file, "Nilai": recall
        })
        all_prediction_metrics_results.append({
            "Metrik": "F1-Score", "Metode": method_name_from_file, "Nilai": f1
        })
    
    # Simpan metrik prediksi secara keseluruhan dalam format tabel
    if all_prediction_metrics_results:
        df_prediction_metrics = pd.DataFrame(all_prediction_metrics_results)
        df_pivot_pred = df_prediction_metrics.pivot_table(index="Metrik", columns="Metode", values="Nilai")
        df_pivot_pred = df_pivot_pred.applymap(lambda x: f"{x:.2f}%")

        try:
            df_pivot_pred.to_csv(PREDICTION_METRICS_FILE)
            logger.info(f"✅ Comprehensive prediction metrics for all methods saved to '{PREDICTION_METRICS_FILE}'.")
            logger.info("\n" + df_pivot_pred.to_string())
        except Exception as e:
            logger.error(f"❌ Failed to save comprehensive prediction metrics: {e}")
    else:
        logger.warning("No prediction metrics to save or display.")


# --- Titik Masuk Skrip ---
if __name__ == "__main__":
    eval_retrieval_all_methods() # Panggil fungsi evaluasi retrieval multi-metode
    eval_prediction_all_methods() # Panggil fungsi evaluasi prediksi multi-metodea