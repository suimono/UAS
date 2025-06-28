import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 
import numpy as np
# from nltk.corpus import stopwords # Uncomment if using Indonesian stopwords


# --- Konfigurasi ---
CASE_BASE_PATH = Path("data/processed/cases.json")
QUERY_PATH = Path("data/eval/queries.json")
OUTPUT_PATH = Path("data/results/retrieved_cases.json")
TOP_K_SIMILAR_CASES = 10 # Meningkatkan K untuk memberikan lebih banyak kandidat ke prediksi, dapat diatur lebih lanjut.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Inisialisasi Model Global (untuk efisiensi) ---
try:
    BERT_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Sentence-BERT model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence-BERT model: {e}. BERT retrieval will not work.")
    BERT_MODEL = None 

# --- Fungsi Utilitas ---

def initialize_directories() -> bool:
    """
    Memastikan direktori keluaran untuk hasil retrieval ada.
    """
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{OUTPUT_PATH.parent}' ensured.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {OUTPUT_PATH.parent}: {e}")
        return False

def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Memuat dan memvalidasi data JSON dari file.
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

def extract_case_text_for_retrieval(case: Dict[str, Any]) -> Optional[str]:
    """
    Mengekstrak teks yang paling relevan dari sebuah kasus untuk tujuan retrieval.
    Prioritas diberikan pada 'ringkasan_fakta', diikuti oleh kombinasi bidang lainnya.
    """
    field_combinations = [
        ["ringkasan_fakta"],
        ["jenis_perkara", "pasal", "status_hukuman"],
        ["jenis_perkara", "pasal"],
        ["no_perkara", "jenis_perkara", "tanggal"],
    ]
    
    for fields_to_try in field_combinations:
        text_parts = []
        for field in fields_to_try:
            if field in case and isinstance(case[field], str):
                value = case[field].strip()
                if (value and 
                    value not in ["===", "---", "...", "N/A", "null", "undefined"] and
                    len(set(value)) > 1 and 
                    len(value) >= 10):
                    
                    if field == "pasal" and len(value) > 200:
                        value = value[:200] + "..."
                    elif field == "status_hukuman" and len(value) > 300:
                        value = value[:300] + "..."
                    
                    text_parts.append(value)
        
        if text_parts:
            return ". ".join(text_parts)
            
    return None

def normalize_scores(scores: List[float]) -> List[float]:
    """Menormalisasi daftar skor ke rentang [0, 1]."""
    if not scores: # Menangani daftar skor kosong
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score: 
        return [0.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

# --- Fungsi Retrieval Spesifik Metode ---

def retrieve_by_tfidf(case_texts: List[str], query_texts: List[str], 
                      case_ids: List[str], query_ids: List[str], 
                      top_k: int = 5) -> Dict[str, Dict[str, List[Any]]]:
    """
    Melakukan retrieval menggunakan metode TF-IDF, mengembalikan ID kasus dan skor kemiripan.
    """
    logger.info("Performing TF-IDF retrieval...")
    
    if not case_texts or not query_texts:
        logger.warning("No case or query texts for TF-IDF retrieval.")
        return {q_id: {"case_ids": [], "scores": []} for q_id in query_ids}

    # Menggunakan stop words Bahasa Indonesia yang lebih baik
    # Pastikan Anda telah menjalankan `nltk.download('stopwords')` jika menggunakan ini.
    # try:
    #     stop_words_list = stopwords.words('indonesian')
    # except LookupError:
    #     logger.warning("NLTK 'stopwords' not found. Please run nltk.download('stopwords'). Using default English stopwords for TF-IDF.")
    #     stop_words_list = 'english'
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) 
    
    corpus = case_texts + query_texts
    tfidf_matrix = vectorizer.fit_transform(corpus)

    case_matrix = tfidf_matrix[:len(case_texts)]
    query_matrix = tfidf_matrix[len(case_texts):]

    similarities = cosine_similarity(query_matrix, case_matrix)

    results = {}
    for i, sim_scores in enumerate(similarities):
        top_indices = sim_scores.argsort()[::-1][:top_k]
        
        top_cases_ids = [case_ids[j] for j in top_indices]
        similarity_scores = [float(sim_scores[j]) for j in top_indices] 
        
        normalized_scores = normalize_scores(similarity_scores)
        
        results[query_ids[i]] = {"case_ids": top_cases_ids, "scores": normalized_scores}
    
    logger.info("TF-IDF retrieval complete.")
    return results

def retrieve_by_bert(case_texts: List[str], query_texts: List[str], 
                     case_ids: List[str], query_ids: List[str], 
                     top_k: int = 5) -> Dict[str, Dict[str, List[Any]]]:
    """
    Melakukan retrieval menggunakan BERT, mengembalikan ID kasus dan skor kemiripan.
    """
    logger.info("Performing BERT retrieval...")
    
    if BERT_MODEL is None:
        logger.error("BERT model not loaded. Skipping BERT retrieval.")
        return {q_id: {"case_ids": [], "scores": []} for q_id in query_ids}

    if not case_texts or not query_texts:
        logger.warning("No case or query texts for BERT retrieval.")
        return {q_id: {"case_ids": [], "scores": []} for q_id in query_ids}

    try:
        case_embeddings = BERT_MODEL.encode(case_texts, show_progress_bar=True, convert_to_tensor=True).cpu().numpy()
        query_embeddings = BERT_MODEL.encode(query_texts, show_progress_bar=True, convert_to_tensor=True).cpu().numpy()

        similarities = cosine_similarity(query_embeddings, case_embeddings)

        results = {}
        for i, sim_scores in enumerate(similarities):
            top_indices = sim_scores.argsort()[::-1][:top_k]
            
            top_cases_ids = [case_ids[j] for j in top_indices]
            similarity_scores = [float(sim_scores[j]) for j in top_indices]
            
            normalized_scores = normalize_scores(similarity_scores)
            
            results[query_ids[i]] = {"case_ids": top_cases_ids, "scores": normalized_scores}
        
        logger.info("BERT retrieval complete.")
        return results
    except Exception as e:
        logger.error(f"Error during BERT retrieval: {e}")
        return {q_id: {"case_ids": [], "scores": []} for q_id in query_ids}


def retrieve_by_hybrid(case_texts: List[str], query_texts: List[str], 
                       case_ids: List[str], query_ids: List[str], 
                       top_k: int = 5) -> Dict[str, Dict[str, List[Any]]]:
    """
    Melakukan retrieval menggunakan metode Hybrid (TF-IDF + BERT), mengembalikan ID kasus dan skor kemiripan.
    Ini menggunakan Weighted Sum of Scores.
    """
    logger.info("Performing Hybrid retrieval (TF-IDF + BERT)...")

    # Dapatkan hasil dari TF-IDF dan BERT terlebih dahulu
    tfidf_results = retrieve_by_tfidf(case_texts, query_texts, case_ids, query_ids, top_k * 2) 
    bert_results = retrieve_by_bert(case_texts, query_texts, case_ids, query_ids, top_k * 2)

    hybrid_results = {}

    for q_id in query_ids:
        combined_scores: Dict[str, float] = {}

        for idx, case_id in enumerate(tfidf_results.get(q_id, {}).get("case_ids", [])):
            score = tfidf_results[q_id]["scores"][idx]
            combined_scores[case_id] = combined_scores.get(case_id, 0.0) + (score * 0.5) 

        for idx, case_id in enumerate(bert_results.get(q_id, {}).get("case_ids", [])):
            score = bert_results[q_id]["scores"][idx]
            combined_scores[case_id] = combined_scores.get(case_id, 0.0) + (score * 0.5) 
        
        sorted_cases = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        
        top_cases_ids = [case_id for case_id, score in sorted_cases[:top_k]]
        similarity_scores = [score for case_id, score in sorted_cases[:top_k]]
        
        hybrid_results[q_id] = {"case_ids": top_cases_ids, "scores": similarity_scores}

    logger.info("Hybrid retrieval complete.")
    return hybrid_results

# --- Fungsi Utama ---

def main():
    """
    Fungsi utama untuk melakukan proses retrieval kasus menggunakan berbagai metode.
    """
    if not initialize_directories():
        return

    # Muat data kasus
    cases = load_json_data(CASE_BASE_PATH)
    if cases is None:
        logger.error("Failed to load case base data. Exiting.")
        return

    # Muat data kueri
    queries = load_json_data(QUERY_PATH)
    if queries is None: # Perbaikan: 'is None' instead of '=== None'
        logger.error("Failed to load queries data. Exiting.")
        return

    if not cases:
        logger.warning("No cases found in the case base. Retrieval cannot proceed.")
        return
    if not queries:
        logger.warning("No queries found. Retrieval cannot proceed.")
        return

    # Ekstrak teks dan ID yang relevan dari kasus dan kueri
    case_ids = []
    case_texts = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            logger.warning(f"Skipping case at index {i}: not a dictionary.")
            continue
        extracted_text = extract_case_text_for_retrieval(case)
        if extracted_text:
            case_ids.append(case.get("case_id", f"case_{i}"))
            case_texts.append(extracted_text)
        else:
            logger.warning(f"Skipping case {case.get('case_id', f'case_{i}')} due to no suitable text content.")

    query_ids = []
    query_texts = []
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            logger.warning(f"Skipping query at index {i}: not a dictionary.")
            continue
        query_text = query.get("text")
        if query_text and isinstance(query_text, str) and query_text.strip():
            query_ids.append(query.get("query_id", f"query_{i}"))
            query_texts.append(query_text.strip())
        else:
            logger.warning(f"Skipping query {query.get('query_id', f'query_{i}')} due to missing or empty 'text' field.")

    if not case_texts:
        logger.error("No valid case texts extracted for retrieval. Cannot proceed.")
        return
    if not query_texts:
        logger.error("No valid query texts extracted for retrieval. Cannot proceed.")
        return
        
    logger.info(f"Extracted {len(case_texts)} valid case texts and {len(query_texts)} valid query texts.")

    # --- Jalankan Setiap Metode Retrieval ---
    retrieval_methods_config = {
        "TF-IDF": retrieve_by_tfidf,
        "BERT": retrieve_by_bert,
        "Hybrid": retrieve_by_hybrid
    }

    all_method_results_raw: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}

    for method_name, retrieve_func in retrieval_methods_config.items():
        method_retrieved_cases_with_scores = retrieve_func(
            case_texts, query_texts, case_ids, query_ids, TOP_K_SIMILAR_CASES
        )
        all_method_results_raw[method_name] = method_retrieved_cases_with_scores

    # --- Gabungkan Hasil ke dalam Struktur Output Baru ---
    final_retrieval_results = []
    for q_id in query_ids:
        query_entry = {
            "query_id": q_id,
            "retrieval_results": {}
        }
        for method_name in retrieval_methods_config.keys():
            retrieved_data_for_method = all_method_results_raw[method_name].get(q_id, {"case_ids": [], "scores": []})
            query_entry["retrieval_results"][method_name] = {
                "case_ids": retrieved_data_for_method["case_ids"],
                "scores": retrieved_data_for_method["scores"]
            }
        final_retrieval_results.append(query_entry)

    # Simpan hasil
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_retrieval_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Retrieval completed. Results for all methods saved to '{OUTPUT_PATH}'")
    except Exception as e:
        logger.error(f"Failed to write retrieval results to '{OUTPUT_PATH}': {e}")

# --- Titik Masuk ---
if __name__ == "__main__":
    main()
