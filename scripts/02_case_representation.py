import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Konfigurasi path
RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/cases.json")
LOG_FILE = Path("data/logs/extraction.log")

def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )

class EnhancedPatternExtractor:
    """Mengelola koleksi pola regex yang digunakan untuk ekstraksi data."""
    def __init__(self):
        # Pattern untuk nomor perkara (Tidak Berubah)
        self.NOMOR_PERKARA_PATTERNS = [
            re.compile(r"PUTUSAN\s+Nomor\s*[:\-]?\s*(\d{1,5}\/[\w\.\-]+?\/\d{4}\/[\w\.]+)", re.IGNORECASE),
            re.compile(r"Nomor\s*[:\-]?\s*(\d{1,5}\/[\w\.\-]+?\/\d{4}\/[\w\.]+)", re.IGNORECASE),
            re.compile(r"No\.\s*(\d{1,5}\/[\w\.\-]+?\/\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,5}\/[\w\.\-]+?\/\d{4}(?:\/[\w\.]+)?)\s*\n", re.IGNORECASE),
            re.compile(r"(\d{1,5}\s*[PKK]{1,2}\/[\w\.\-]+?\/\d{4})", re.IGNORECASE)
        ]
        
        # Pattern untuk tanggal (Tidak Berubah)
        self.DATE_PATTERNS = [
            re.compile(r"(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(\d{4})", re.IGNORECASE),
            re.compile(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})"),
            re.compile(r"(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})"),
            re.compile(r"tanggal\s+(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE),
            re.compile(r"pada\s+hari\s+\w+\s+tanggal\s+(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE)
        ]
        
        # Pattern untuk jenis perkara (Tidak Berubah)
        self.JENIS_PERKARA_PATTERNS = [
            re.compile(r"Tindak\s+Pidana\s+Korupsi", re.IGNORECASE),
            re.compile(r"Tipikor", re.IGNORECASE),
            re.compile(r"Narkotika", re.IGNORECASE),
            re.compile(r"Pidana\s+Khusus", re.IGNORECASE),
            re.compile(r"Pidana\s+Umum", re.IGNORECASE),
            re.compile(r"Perdata", re.IGNORECASE),
            re.compile(r"Tata\s+Usaha\s+Negara", re.IGNORECASE),
            re.compile(r"TUN", re.IGNORECASE),
            re.compile(r"PHI", re.IGNORECASE),
            re.compile(r"Perkawinan", re.IGNORECASE),
            re.compile(r"Waris", re.IGNORECASE),
            re.compile(r"Ekonomi\s+Syariah", re.IGNORECASE),
            re.compile(r"Gugatan\s+Sederhana", re.IGNORECASE),
            re.compile(r"Lalu\s+Lintas", re.IGNORECASE),
            re.compile(r"Ketenagakerjaan", re.IGNORECASE)
        ]
        
        # Pattern untuk pasal (Tidak Berubah)
        self.PASAL_PATTERNS = [
            re.compile(
                r"(?:terbukti|bersalah|melakukan\s+tindak\s+pidana|dihukum)\s+.*?"
                r"(?:sebagaimana\s+diatur\s+dalam|melanggar|berdasarkan)\s+"
                r"((?:Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)"
                r"(?:[\s\.\,\;]*(?:jo\.?|juncto|dan|atau|serta)?\s*Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)*)"
                r"(?:\s+Undang-Undang\s+Nomor\s+\d+)?",
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(
                r"(?:menyatakan|memutuskan|menimbang|mengadili|berdasarkan)\s+.*?"
                r"((?:Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)"
                r"(?:[\s\.\,\;]*(?:jo\.?|juncto|dan|atau|serta)?\s*Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)*)"
                r"(?:\s+Undang-Undang\s+Nomor\s+\d+)?",
                re.IGNORECASE | re.DOTALL
            ),
            re.compile(r"((?:Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)(?:[\s\.\,\;]*(?:jo\.?|juncto|dan|atau|serta)?\s*Pasal\s+\d+(?:\s+Ayat\s*\(\d+\))?(?:\s+huruf\s+[a-zA-Z])?)*)", re.IGNORECASE)
        ]
        
        # --- PERBAIKAN DI SINI: Pattern untuk data personal (nama) ---
        # Pola disempurnakan untuk lebih akurat menangkap nama dan memfilter noise
        self.PERSONAL_PATTERNS = {
            'nama': [
                # Pola terkuat: Nama yang diikuti oleh info identitas (TTL, Umur, JK, Pekerjaan, Alamat)
                # Menangkap nama, opsional gelar/peran di depannya, lalu nama utama, opsional bin/binti.
                re.compile(r"(?:Terdakwa|Penggugat|Tergugat|Pemohon|Pembanding|Terbanding|Kuasa Hukum|Jaksa Penuntut Umum|Penasehat Hukum|Saksi|Ahli)?\s*[:\-\,\.\(\)\s]*([A-Za-z][a-zA-Z\s\.\-']{2,80}(?:\s+(?:bin|binti)\s+[A-Za-z][a-zA-Z\s\.\-']{2,80})?)(?:\s*,)?\s*(?:Tempat\s+lahir|TTL|lahir|Umur|Usia|Jenis\s+Kelamin|Pekerjaan|Alamat|Pekerjaan|Jabatan)", re.IGNORECASE),
                # Nama Lengkap: [Nama] atau Nama : [Nama]
                re.compile(r"(?:Nama|Nama Lengkap)\s*[:\-]?\s*([A-Za-z][a-zA-Z\s\.\-']{2,80}(?:\s+(?:bin|binti)\s+[A-Za-z][a-zA-Z\s\.\-']{2,80})?)", re.IGNORECASE),
                # Peran: [Terdakwa/Penggugat/dll.]: [Nama]
                re.compile(r"(?:Terdakwa|Penggugat|Tergugat|Pemohon|Pembanding|Terbanding|Pemohon Kasasi|Termohon Kasasi|Jaksa Penuntut Umum)\s*[:\-]?\s*([A-Za-z][a-zA-Z\s\.\-']{2,80}(?:\s+(?:bin|binti)\s+[A-Za-z][a-zA-Z\s\.\-']{2,80})?)", re.IGNORECASE),
                # a.n. [Nama] atau oleh: [Nama]
                re.compile(r"(?:a\.n\.|oleh)\s*:\s*([A-Za-z][a-zA-Z\s\.\-']{2,80}(?:\s+(?:bin|binti)\s+[A-Za-z][a-zA-Z\s\.\-']{2,80})?)", re.IGNORECASE),
                # Menyatakan/Menjatuhkan pidana kepada Terdakwa [Nama]
                re.compile(r"(?:menyatakan|menjatuhkan)\s+(?:pidana|hukuman)\s+kepada\s+(?:Terdakwa|Para Terdakwa|Anak)\s+([A-Za-z][a-zA-Z\s\.\-']{2,80}(?:\s+(?:bin|binti)\s+[A-Za-z][a-zA-Z\s\.\-']{2,80})?)", re.IGNORECASE)
            ],
            'umur': [
                re.compile(r"Umur[\/\s]*Tanggal\s*lahir\s*[:\-]?\s*(\d{1,3})\s*(?:tahun|thn)", re.IGNORECASE),
                re.compile(r"Umur\s*[:\-]?\s*(\d{1,3})\s*(?:tahun|thn)", re.IGNORECASE)
            ],
            'jenis_kelamin': [
                re.compile(r"Jenis\s+Kelamin\s*[:\-]?\s*(Laki-laki|Perempuan|L|P)\b", re.IGNORECASE),
                re.compile(r"Kelamin\s*[:\-]?\s*(Laki-laki|Perempuan|L|P)\b", re.IGNORECASE)
            ],
            'pekerjaan': [
                re.compile(r"Pekerjaan\s*[:\-]?\s*([^:\n]{3,60})", re.IGNORECASE),
                re.compile(r"Jabatan\s*[:\-]?\s*([^:\n]{3,60})", re.IGNORECASE)
            ],
            'alamat': [
                re.compile(r"(?:Tempat\s+Tinggal|Alamat)\s*[:\-]?\s*([^:\n]{10,250}\.?\s*(?:RT|RW|No|Jalan|Kelurahan|Kecamatan|Kota|Kabupaten|Provinsi)\s*[^:\n]{5,100})?", re.IGNORECASE),
                re.compile(r"beralamat\s+di\s+([^:\n]{10,250}\.?\s*(?:RT|RW|No|Jalan|Kelurahan|Kecamatan|Kota|Kabupaten|Provinsi)\s*[^:\n]{5,100})?", re.IGNORECASE)
            ]
        }

        self.STATUS_HUKUMAN_PATTERNS = [
            re.compile(r'(?:menyatakan|mengadili).*?(?:terbukti|bersalah|tidak\s+terbukti|bebas|dihukum|dipidana).*?dengan\s+pidana\s+([^.\n]{20,300}\.?)(?:\n|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:menyatakan|memutuskan|mengadili).*?(?:terbukti\s+secara\s+sah\s+dan\s+meyakinkan|bersalah|tidak\s+terbukti|bebas)[^\.]*\.?', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:pidana|hukuman).*?(?:penjara|denda|kurungan|rehabilitasi|bebas).*?[^\.]*\.?', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?:terdakwa|pemohon).*?(?:dipidana|dijatuhi|dihukum).*?[^\.]*\.?', re.IGNORECASE | re.DOTALL)
        ]

class ImprovedSmartExtractor:
    """
    Kelas untuk mengekstrak metadata terstruktur dari teks dokumen hukum mentah.
    """
    def __init__(self):
        self.patterns = EnhancedPatternExtractor()
        self.logger = logging.getLogger(__name__)
        
        self.BULAN_MAP = {
            'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
            'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
        }

    def clean_text(self, text: str) -> str:
        """Bersihkan teks dari karakter tidak perlu (Tidak Berubah)."""
        if not text:
            return ""
        
        text = text.replace('\x00', ' ').replace('\xa0', ' ')
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s*\(\s*', '(', text)
        text = re.sub(r'\s*\)\s*', ')', text)
        return text.strip()

    def _find_first_valid_match(self, text: str, patterns: List[re.Pattern], search_limit: int) -> str:
        """Mencari dan mengembalikan match pertama yang valid (Tidak Berubah)."""
        search_area = text[:search_limit]
        for pattern in patterns:
            match = pattern.search(search_area)
            if match:
                result = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                result = re.sub(r'\s+', ' ', result).strip()
                if result:
                    return result
        return ""

    def extract_nomor_perkara(self, text: str) -> str:
        """Ekstrak nomor perkara (Tidak Berubah)."""
        nomor = self._find_first_valid_match(text, self.patterns.NOMOR_PERKARA_PATTERNS, 5000)
        nomor = nomor.replace("Nomor :", "").replace("No.", "").strip()
        return nomor

    def extract_tanggal(self, text: str) -> str:
        """Ekstrak tanggal (Tidak Berubah)."""
        search_area = text[:8000] 
        
        for pattern in self.patterns.DATE_PATTERNS:
            matches = pattern.finditer(search_area)
            for match in matches:
                start_pos = max(0, match.start() - 150)
                end_pos = min(len(search_area), match.end() + 150)
                context = search_area[start_pos:end_pos].lower()
                
                if any(keyword in context for keyword in ['lahir', 'usia', 'umur', 'ktp', 'identitas', 'akta', 'ijazah']):
                    self.logger.debug(f"Skipping date '{match.group(0)}' due to context: {context[:50]}...")
                    continue
                
                day, month, year = None, None, None
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                elif len(match.groups()) == 4:
                    day, month, year = match.group(1), match.group(2), match.group(3)
                
                if day and month and year:
                    if month.lower() in self.BULAN_MAP:
                        month_str = self.BULAN_MAP[month.lower()]
                    elif month.isdigit() and 1 <= int(month) <= 12:
                        month_str = f"{int(month):02d}"
                    else:
                        continue

                    try:
                        day_num = int(day)
                        year_num = int(year)
                        month_num = int(month_str)
                        
                        if 1 <= day_num <= 31 and 1 <= month_num <= 12 and 1990 <= year_num <= datetime.now().year + 5:
                            return f"{year_num}-{month_str}-{int(day_num):02d}"
                    except ValueError:
                        continue
        return ""

    def extract_jenis_perkara(self, text: str) -> str:
        """Ekstrak jenis perkara (Tidak Berubah)."""
        jenis = self._find_first_valid_match(text, self.patterns.JENIS_PERKARA_PATTERNS, 5000)
        if jenis:
            return jenis.title()
        
        text_lower = text.lower()
        if re.search(r'tindak\s+pidana\s+korupsi|korupsi|suap|gratifikasi|tipikor', text_lower):
            return "Tindak Pidana Korupsi"
        elif re.search(r'narkoba|narkotika|psikotropika', text_lower):
            return "Narkotika"
        elif re.search(r'pidana\s+khusus|pid.sus', text_lower):
            return "Pidana Khusus"
        elif re.search(r'pidana\s+umum|pid.umum', text_lower):
            return "Pidana Umum"
        elif re.search(r'perdata|pdt', text_lower):
            return "Perdata"
        elif re.search(r'tata\s+usaha\s+negara|tun', text_lower):
            return "Tata Usaha Negara"
        
        return ""

    def extract_pasal(self, text: str) -> List[str]:
        """Ekstrak pasal-pasal yang dilanggar (Tidak Berubah, sudah diperbaiki di iterasi sebelumnya)."""
        pasal_found = set()
        
        for pattern in self.patterns.PASAL_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                pasal_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                
                pasal_text = re.sub(r'\s+', ' ', pasal_text)
                pasal_text = re.sub(r'Ayat\s*\(\s*(\d+)\s*\)', r'Ayat (\1)', pasal_text, re.IGNORECASE)
                pasal_text = re.sub(r'huruf\s*([a-zA-Z])', r'huruf \1', pasal_text, re.IGNORECASE)
                pasal_text = pasal_text.replace("jo.", "jo").replace("juncto", "jo").replace("serta", "dan").replace("atau", "dan")

                sub_pasals = re.split(r'\s+(?:jo|dan)\s+', pasal_text, flags=re.IGNORECASE)
                
                for p in sub_pasals:
                    p = p.strip()
                    if re.search(r'Pasal\s+\d+', p, re.IGNORECASE):
                        if 5 <= len(p) <= 150:
                            pasal_found.add(p.title())

        return sorted(list(pasal_found))

    # --- PERBAIKAN DI SINI: extract_personal_data (untuk nama) ---
    def extract_personal_data(self, text: str, field: str) -> str:
        """Ekstrak data personal berdasarkan field dengan validasi lebih ketat."""
        if field not in self.patterns.PERSONAL_PATTERNS:
            return ""
        
        search_area = text[:20000] # Perluas area pencarian untuk data personal
        
        best_match_value = "" # Untuk menyimpan kandidat nama terbaik
        
        for pattern in self.patterns.PERSONAL_PATTERNS[field]:
            matches = pattern.findall(search_area)
            if matches:
                for match in matches:
                    # Ambil group pertama jika itu tuple (dari capture group), atau match langsung
                    value = match[0].strip() if isinstance(match, tuple) else match.strip()
                    
                    # Pembersihan umum dari karakter yang tidak diinginkan dalam nama/teks
                    value = re.sub(r'[^\w\s\.\-,\(\)/]', '', value).strip()
                    value = re.sub(r'\s+', ' ', value).strip() # Normalisasi spasi

                    # Validasi spesifik per bidang
                    if field == 'umur':
                        if value.isdigit() and 10 <= int(value) <= 100:
                            return value
                    elif field == 'jenis_kelamin':
                        if value.lower() in ['laki-laki', 'perempuan', 'l', 'p']:
                            return "Laki-laki" if value.lower() in ['laki-laki', 'l'] else "Perempuan"
                    elif field == 'nama':
                        # Kriteria validasi nama
                        if len(value) < 3 or not re.search(r'[a-zA-Z]', value): # Harus punya huruf dan cukup panjang
                            continue
                        
                        # Filter nama yang terlalu umum atau bukan nama individu
                        excluded_terms = ["terdakwa", "penggugat", "tergugat", "pemohon", "kuasa hukum", "majelis hakim", 
                                          "saksi", "ahli", "jaksa penuntut umum", "panitera", "hakim ketua", 
                                          "hakim anggota", "panitera pengganti", "para terdakwa", "para penggugat"]
                        if value.lower() in excluded_terms or any(term in value.lower() for term in excluded_terms):
                            continue

                        # Jika nama mengandung "bin" atau "binti", itu indikator kuat nama lengkap
                        if re.search(r'\b(bin|binti)\b', value, re.IGNORECASE) and len(value.split()) >= 3:
                            # Prioritas tinggi: jika ini ditemukan, langsung kembalikan
                            return value.title() 

                        # Cek apakah nama terdiri dari 2 kata atau lebih dan setiap kata dimulai kapital
                        # Contoh: "Budi Santoso", "Dr. A. Yani"
                        if re.match(r"^(?:[A-Z][a-zA-Z\.\-']+\s+){1,}[A-Z][a-zA-Z\.\-']+$", value):
                            # Jika ini lebih panjang atau lebih spesifik dari kandidat sebelumnya
                            if len(value) > len(best_match_value):
                                best_match_value = value.title() # Simpan sebagai kandidat terbaik
                            
                        # Fallback untuk nama satu kata kapital atau dua kata dengan kapital di awal saja
                        elif len(value.split()) >= 1 and re.match(r"^[A-Z][a-zA-Z\s\.\-']+$", value):
                             if len(value) > len(best_match_value): # Simpan jika lebih panjang
                                best_match_value = value.title()
                        
                        # Jika sudah menemukan kandidat yang cukup baik, bisa langsung break
                        if best_match_value and len(best_match_value.split()) > 1: # Minimal 2 kata
                            break # Hentikan pencarian pola lain untuk nama ini
                        
                    elif field == 'pekerjaan' and len(value) >= 3 and len(value) <= 60:
                        return value
                    elif field == 'alamat' and len(value) >= 20 and len(value) <= 250:
                        if any(k in value.lower() for k in ['jalan', 'no', 'rt', 'rw', 'kelurahan', 'kecamatan', 'kota', 'kabupaten']):
                            return value
                
        # Jika loop selesai, kembalikan kandidat nama terbaik yang ditemukan
        return best_match_value

    def extract_status_hukuman(self, text: str) -> str:
        """Ekstrak status hukuman/putusan (Sudah diperbaiki dari error sebelumnya)."""
        search_area = text[-7000:] if len(text) > 7000 else text
        
        # Menggunakan pola yang sudah dikompilasi dari EnhancedPatternExtractor
        for pattern in self.patterns.STATUS_HUKUMAN_PATTERNS:
            matches = re.findall(pattern, search_area)
            for match in matches:
                status_text = match[0].strip() if isinstance(match, tuple) else match.strip()
                status_text = re.sub(r'\s+', ' ', status_text).strip()
                
                if 30 <= len(status_text) <= 500:
                    return status_text
            
        return ""

    def extract_ringkasan_fakta(self, text: str, min_len: int = 200, max_len: int = 1500) -> str:
        """
        Ekstrak ringkasan fakta dari dokumen, membersihkan header/footer,
        dan memastikan panjang yang wajar.
        """
        # Bersihkan teks dasar
        text = self.clean_text(text)
        if not text:
            return ""

        # Pola untuk menandai awal dan akhir potensial dari ringkasan fakta
        start_keywords = [
            re.compile(r'DUDUK\s+PERKARA', re.IGNORECASE),
            re.compile(r'I\.\s*PERKARA', re.IGNORECASE),
            re.compile(r'FAKTA-FAKTA', re.IGNORECASE),
            re.compile(r'MENIMBANG(?:\s+BAHWA|\,\s+bahwa)?\s+permohonan', re.IGNORECASE),
            re.compile(r'DALAM\s+POKOK\s+PERKARA', re.IGNORECASE),
            re.compile(r'TENTANG\s+PERKARA', re.IGNORECASE),
            re.compile(r'POKOK\s+GUGATAN', re.IGNORECASE), # Untuk perdata/TUN
            re.compile(r'URAIAN\s+PERBUATAN', re.IGNORECASE), # Untuk pidana
        ]
        
        end_keywords = [
            re.compile(r'TENTANG\s+HUKUM', re.IGNORECASE),
            re.compile(r'MENIMBANG(?:\s+TENTANG|\,\s+tentang)?\s+HUKUM', re.IGNORECASE),
            re.compile(r'DALAM\s+PERTIMBANGAN\s+HUKUM', re.IGNORECASE),
            re.compile(r'MEMUTUSKAN', re.IGNORECASE),
            re.compile(r'MENGADILI', re.IGNORECASE),
            re.compile(r'AMAR\s+PUTUSAN', re.IGNORECASE),
            re.compile(r'DALAM\s+EKSEPSI', re.IGNORECASE), # Jika ada eksepsi setelah fakta
            re.compile(r'Demikian\s+diputus\s+dalam\s+rapat\s+musyawarah', re.IGNORECASE),
        ]

        fact_start_pos = -1
        fact_end_pos = -1

        # Cari awal bagian fakta
        for pattern in start_keywords:
            match = pattern.search(text)
            if match:
                fact_start_pos = match.end()
                break

        # Cari akhir bagian fakta (mulai pencarian dari fact_start_pos jika ditemukan)
        search_from = fact_start_pos if fact_start_pos != -1 else 0
        for pattern in end_keywords:
            match = pattern.search(text, search_from)
            if match:
                fact_end_pos = match.start()
                break

        extracted_content = ""
        if fact_start_pos != -1 and fact_end_pos != -1 and fact_end_pos > fact_start_pos:
            extracted_content = text[fact_start_pos:fact_end_pos].strip()
            self.logger.debug(f"Fact section found between {fact_start_pos} and {fact_end_pos}")
        elif fact_start_pos != -1: # Jika hanya awal ditemukan, ambil dari awal hingga akhir dokumen
            extracted_content = text[fact_start_pos:].strip()
            self.logger.debug(f"Fact section from {fact_start_pos} to end (no end marker).")
        elif fact_end_pos != -1: # Jika hanya akhir ditemukan, ambil dari awal dokumen hingga posisi akhir
            extracted_content = text[:fact_end_pos].strip()
            self.logger.debug(f"Fact section from beginning to {fact_end_pos} (no start marker).")
        else: # Fallback: tidak ada penanda jelas, coba ekstrak dari blok teks substansial
            self.logger.debug("No clear fact markers found. Using heuristic fallback.")
            # Hapus header umum yang sangat mungkin ada di awal dokumen
            text_after_header_clean = re.sub(
                r'^(?:PUTUSAN\s+NOMOR\s+[\s\S]*?DENGAN\s+RAHMAT\s+TUHAN\s+YANG\s+MAHA\s+ESA|PENGADILAN\s+NEGERI.*?\n+)*',
                '', text, flags=re.IGNORECASE | re.DOTALL
            ).strip()

            # Ambil beberapa paragraf pertama yang substansial
            lines = text_after_header_clean.split('\n')
            
            # Filter baris yang sangat pendek atau hanya berisi penomoran/artefak
            content_candidate_lines = []
            for line in lines:
                stripped_line = line.strip()
                # Hindari baris yang terlalu pendek, nomor halaman, atau baris yang terlalu berulang
                if len(stripped_line) > 50 and not re.fullmatch(r'^\s*[-_]?\s*\d+\s*[-_]?\s*$', stripped_line) and \
                   not re.fullmatch(r'^\s*[A-Z\s]+\s*$', stripped_line) and \
                   not re.fullmatch(r'^[\s\W_]*$', stripped_line): # Bukan hanya simbol
                    content_candidate_lines.append(stripped_line)
                elif len(stripped_line) > 10 and re.search(r'[a-zA-Z0-9]', stripped_line): # Keep reasonably long lines with words
                    content_candidate_lines.append(stripped_line)

            extracted_content = "\n".join(content_candidate_lines).strip()
            self.logger.debug(f"Fallback extracted content length: {len(extracted_content)}")

        # --- Pembersihan akhir dan pemotongan panjang ---
        extracted_content = self.clean_text(extracted_content) # Bersihkan lagi setelah segmentasi
        
        # Hapus sisa-sisa pola umum yang tidak diinginkan, tapi dengan hati-hati
        general_cleanup_patterns = [
            re.compile(r'Disclaimer\s*[:\-].*$', re.IGNORECASE | re.DOTALL),
            re.compile(r'Halaman\s+\d+\s+dari\s+\d+.*$', re.IGNORECASE | re.DOTALL),
            re.compile(r'MAHKAMAH\s+AGUNG.*$', re.IGNORECASE | re.DOTALL),
            re.compile(r'Kepaniteraan.*@mahkamahagung\.go\.id.*$', re.IGNORECASE | re.DOTALL),
            re.compile(r'Catatan\s*:\s*Putusan\s*ini.*$', re.IGNORECASE | re.DOTALL),
            re.compile(r'\bSALINAN\b[\s\S]*?\bPANITERA\b', re.IGNORECASE | re.DOTALL),
        ]
        for pattern in general_cleanup_patterns:
            extracted_content = re.sub(pattern, '', extracted_content) # Apply already compiled patterns
        extracted_content = self.clean_text(extracted_content) # Bersihkan final

        # Potong sesuai panjang maksimal
        if len(extracted_content) > max_len:
            extracted_content = extracted_content[:max_len]
            last_period = extracted_content.rfind('.')
            if last_period > min_len: # Pastikan tidak memotong terlalu awal dari min_len
                extracted_content = extracted_content[:last_period + 1]
            else: # Jika tidak ada titik yang baik, potong di kata terakhir
                extracted_content = extracted_content.rsplit(' ', 1)[0] + "..."
        
        # Pastikan panjang minimum terpenuhi, jika tidak, coba lebih agresif dari teks asli
        if len(extracted_content) < min_len and len(text) >= min_len:
            self.logger.warning(f"Extracted facts too short ({len(extracted_content)} chars). Original text had {len(text)} chars. Attempting aggressive fallback.")
            # Ambil bagian awal teks setelah header yang sangat umum (lebih berani)
            aggressive_fallback_text = re.sub(
                r'^(.*?PUTUSAN\s+NOMOR\s+[\s\S]*?(?:DENGAN\s+RAHMAT\s+TUHAN\s+YANG\s+MAHA\s+ESA|MAJELIS\s+HAKIM|MENIMBANG|MENGADILI)\s*?\n+)?',
                '', text, flags=re.IGNORECASE | re.DOTALL
            )
            aggressive_fallback_text = self.clean_text(aggressive_fallback_text)
            if len(aggressive_fallback_text) > min_len:
                return aggressive_fallback_text[:min_len].strip() + "..."
            else:
                return aggressive_fallback_text.strip() # Kembalikan apa pun yang tersisa

        return extracted_content.strip()

    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Ekstrak semua metadata dari teks (Tidak Berubah)."""
        if not text:
            self.logger.warning("Input text is empty, returning empty metadata.")
            return {}
        
        text = self.clean_text(text)
        
        metadata = {
            "no_perkara": self.extract_nomor_perkara(text),
            "tanggal": self.extract_tanggal(text),
            "jenis_perkara": self.extract_jenis_perkara(text),
            "pasal": "; ".join(self.extract_pasal(text)),
            "nama": self.extract_personal_data(text, 'nama'),
            "umur": self.extract_personal_data(text, 'umur'),
            "jenis_kelamin": self.extract_personal_data(text, 'jenis_kelamin'),
            "pekerjaan": self.extract_personal_data(text, 'pekerjaan'),
            "alamat": self.extract_personal_data(text, 'alamat'),
            "status_hukuman": self.extract_status_hukuman(text),
            "ringkasan_fakta": self.extract_ringkasan_fakta(text)
        }
        
        found_fields = [k for k, v in metadata.items() if v and v.strip() and v not in ["N/A", "UNKNOWN"]]
        self.logger.debug(f"Extracted fields: {found_fields}")
        
        return metadata

def process_all_cases():
    """Proses semua file txt dalam folder raw dan simpan sebagai JSON (Tidak Berubah)."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {RAW_DIR}. Please ensure raw text files are in this directory.")
        print(f"❌ Tidak ada file .txt ditemukan di {RAW_DIR}")
        return
    
    logger.info(f"Found {len(txt_files)} .txt files to process.")
    print(f"🔍 Ditemukan {len(txt_files)} file untuk diproses.")
    
    extractor = ImprovedSmartExtractor()
    results = []
    success_count = 0
    error_count = 0
    
    for file_path in txt_files:
        current_case_id = file_path.stem
        try:
            logger.info(f"Processing {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"File {file_path.name} is empty. Skipping.")
                continue
            
            metadata = extractor.extract_metadata(text)
            
            if metadata.get("no_perkara") and len(metadata["no_perkara"]) > 10:
                current_case_id = metadata["no_perkara"].replace("/", "_").replace(".", "_").strip()
            
            metadata.update({
                "case_id": current_case_id,
                "file_name": file_path.name,
                "file_size": len(text),
                "processed_at": datetime.now().isoformat()
            })
            
            results.append(metadata)
            success_count += 1
            
            found_count = sum(1 for k, v in metadata.items() if k not in ['case_id', 'file_name', 'file_size', 'processed_at'] and v and v.strip() and v not in ["N/A", "UNKNOWN"])
            print(f"✅ {file_path.name} (ID: {current_case_id}) - {found_count} field terisi.")
            
        except Exception as e:
            error_count += 1
            logger.error(f"Failed to process {file_path.name} (ID: {current_case_id}): {e}", exc_info=True)
            print(f"❌ Error processing {file_path.name} (ID: {current_case_id}): {e}")
    
    if results:
        unique_case_ids = set()
        cleaned_results = []
        for res in results:
            if res['case_id'] in unique_case_ids:
                logger.warning(f"Duplicate case_id '{res['case_id']}' found for file '{res['file_name']}'. Skipping this duplicate entry.")
            else:
                unique_case_ids.add(res['case_id'])
                cleaned_results.append(res)
        
        if len(cleaned_results) < len(results):
            logger.warning(f"Removed {len(results) - len(cleaned_results)} duplicate case entries.")
            results = cleaned_results

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(results)} unique cases to {OUTPUT_FILE}.")
        print(f"💾 Berhasil menyimpan {len(results)} kasus unik ke {OUTPUT_FILE}.")
        
        print(f"\n📊 STATISTIK EKSTRAKSI KESELURUHAN:")
        print(f"Total file .txt ditemukan: {len(txt_files)}")
        print(f"Berhasil diproses (unik): {success_count - (len(cleaned_results) - len(results))}")
        print(f"Error saat proses file: {error_count}")
        print(f"Total kasus disimpan di '{OUTPUT_FILE}': {len(results)}")
        
        field_stats = {}
        expected_fields = [
            "no_perkara", "tanggal", "jenis_perkara", "pasal", "nama", 
            "umur", "jenis_kelamin", "pekerjaan", "alamat", "status_hukuman", "ringkasan_fakta"
        ]
        for field in expected_fields:
            field_stats[field] = 0

        for result in results:
            for field in expected_fields:
                if field in result and result[field] and result[field].strip() and result[field] not in ["N/A", "UNKNOWN"]:
                    field_stats[field] += 1
        
        print(f"\n📈 REKAP EKSTRAKSI PER FIELD (dari {len(results)} kasus):")
        for field, count in sorted(field_stats.items()):
            percentage = (count / len(results)) * 100
            print(f"- {field.replace('_', ' ').title()}: {count}/{len(results)} ({percentage:.1f}%)")
    
    else:
        logger.warning("No cases were successfully processed or saved.")
        print("❌ Tidak ada kasus yang berhasil diproses atau disimpan.")

if __name__ == "__main__":
    process_all_cases()
