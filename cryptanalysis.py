import os
import time
import random
import string
import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crypto
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# Try to import llama_cpp; if missing, we'll use a simulated classifier
LLAMA_AVAILABLE = True
try:
    from llama_cpp import Llama
except Exception as e:
    LLAMA_AVAILABLE = False

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "codellama-7b.Q5_K_M.gguf"   # your GGUF file
LLAMA_THREADS = 8
USE_LLAMA = True                           # set False to force simulated classifier
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
sns.set_theme(style="whitegrid", font_scale=1.05)

# ----------------------------
# Utilities & classical ciphers
# ----------------------------
ALPHABET = string.ascii_uppercase
BLOCK = 16

def normalize_text(s: str) -> str:
    s = ''.join(ch for ch in s.upper() if ch.isalpha() or ch.isspace())
    return ' '.join(s.split())

# Caesar
def caesar_encrypt(plain: str, k: int) -> str:
    out = []
    for ch in plain:
        if ch.isalpha():
            out.append(chr((ord(ch)-65 + k) % 26 + 65))
        else:
            out.append(ch)
    return ''.join(out)

def caesar_decrypt(ct: str, k: int) -> str:
    return caesar_encrypt(ct, (-k) % 26)

# Affine
def _inv_mod_26(a: int) -> Optional[int]:
    for i in range(26):
        if (a * i) % 26 == 1:
            return i
    return None

def affine_encrypt(plain: str, a: int, b: int) -> str:
    out=[]
    for ch in plain:
        if ch.isalpha():
            x = ord(ch)-65
            out.append(chr((a*x + b) % 26 + 65))
        else:
            out.append(ch)
    return ''.join(out)

def affine_decrypt(ct: str, a:int, b:int) -> str:
    a_inv = _inv_mod_26(a)
    if a_inv is None:
        raise ValueError("No modular inverse for 'a'")
    out=[]
    for ch in ct:
        if ch.isalpha():
            y = ord(ch)-65
            x = (a_inv * (y - b)) % 26
            out.append(chr(x+65))
        else:
            out.append(ch)
    return ''.join(out)

# Vigenere
def vigenere_encrypt(plain: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())
    out=[]; ki=0
    for ch in plain:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65
            out.append(chr((ord(ch)-65 + k) % 26 + 65))
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)

def vigenere_decrypt(ct: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())
    out=[]; ki=0
    for ch in ct:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65
            out.append(chr((ord(ch)-65 - k) % 26 + 65))
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)

# English frequency scoring (for classical solvers)
ENGLISH_FREQ = {
 'A':0.08167,'B':0.01492,'C':0.02782,'D':0.04253,'E':0.12702,'F':0.02228,'G':0.02015,
 'H':0.06094,'I':0.06966,'J':0.00153,'K':0.00772,'L':0.04025,'M':0.02406,'N':0.06749,
 'O':0.07507,'P':0.01929,'Q':0.00095,'R':0.05987,'S':0.06327,'T':0.09056,'U':0.02758,
 'V':0.00978,'W':0.02360,'X':0.00150,'Y':0.01974,'Z':0.00074
}

def english_score(text:str)->float:
    t = ''.join(ch for ch in text.upper() if ch.isalpha())
    N = len(t)
    if N == 0:
        return -9999.0
    freqs = Counter(t)
    score = 0.0
    for ch, f in ENGLISH_FREQ.items():
        obs = freqs.get(ch, 0) / N
        score -= (obs - f) ** 2
    return score

# Classical solvers
def best_caesar(ct: str) -> Tuple[int,str,float]:
    best_s = -1e9; best_k=None; best_p=None
    for k in range(26):
        p = caesar_decrypt(ct, k)
        s = english_score(p)
        if s > best_s:
            best_s, best_k, best_p = s, k, p
    return best_k, best_p, best_s

def best_affine(ct: str) -> Tuple[Tuple[int,int],str,float]:
    best_s = -1e9; best_key=None; best_p=None
    for a in range(1,26):
        if math.gcd(a, 26) != 1:
            continue
        for b in range(26):
            p = affine_decrypt(ct, a, b)
            s = english_score(p)
            if s > best_s:
                best_s, best_key, best_p = s, (a,b), p
    return best_key, best_p, best_s

def index_of_coincidence(text: str) -> float:
    t = ''.join(ch for ch in text if ch.isalpha())
    N = len(t)
    if N <= 1: return 0.0
    freqs = Counter(t)
    return sum(v*(v-1) for v in freqs.values())/(N*(N-1))

def friedman_estimate(ct: str) -> int:
    ic = index_of_coincidence(ct)
    if ic <= 0:
        return 1
    k = (0.0265 * len(ct)) / ((0.065 - ic) + (len(ct) * (ic - 0.0385)))
    return max(1, int(round(k)))

def vigenere_attack(ct: str, max_len: int=10) -> Tuple[str,str,float]:
    guess_len = min(max_len, max(1, friedman_estimate(ct)))
    key = ''
    for i in range(guess_len):
        subseq = ''.join(ch for idx,ch in enumerate(ct) if ch.isalpha() and (idx % guess_len) == i)
        best_shift, best_score = 0, -1e9
        for s in range(26):
            dec = ''.join(chr((ord(ch)-65 - s) % 26 + 65) for ch in subseq)
            sc = english_score(dec)
            if sc > best_score:
                best_shift, best_score = s, sc
        key += chr(best_shift + 65)
    plain = vigenere_decrypt(ct, key)
    return key, plain, english_score(plain)

# ----------------------------
# AES helpers and detectors
# ----------------------------
def aes_ecb_encrypt_bytes(pt: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(pt, BLOCK))

def aes_cbc_encrypt_bytes(pt: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.encrypt(pad(pt, BLOCK))

def aes_ctr_encrypt_bytes(pt: bytes, key: bytes, nonce: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return cipher.encrypt(pt)  # CTR doesn't require padding

def detect_ecb_repetition(ct: bytes, block_size:int=16) -> bool:
    blocks = [ct[i:i+block_size] for i in range(0, len(ct), block_size)]
    return len(set(blocks)) < len(blocks)

def detect_all_zero_iv(iv: Optional[bytes]) -> bool:
    if iv is None:
        return False
    return all(b==0 for b in iv)

def detect_padding_like_signature(ct: bytes) -> bool:
    # purely heuristic: ends with a full block of same pad byte (fake signal)
    if len(ct) < BLOCK: return False
    last = ct[-1]
    return ct.endswith(bytes([last]) * last) if 1 <= last <= BLOCK else False

# ----------------------------
# Toy repeated-byte brute-force (SAFE demo only)
# ----------------------------
def toy_repeated_byte_bruteforce(ct_bytes: bytes, iv: Optional[bytes], mode: str, max_candidates:int=256):
    """
    Educational-only brute force for toy keys of the form byte * 16.
    Safe default = 256 candidates (0..255).
    The function refuses to run when max_candidates is huge.
    """
    if max_candidates > 2**20:
        raise RuntimeError("Refusing to brute-force a large keyspace for safety.")
    found = []
    attempts = 0
    for b in range(max_candidates):
        attempts += 1
        k = bytes([b]) * 16
        try:
            if mode == 'ECB':
                plain_padded = AES.new(k, AES.MODE_ECB).decrypt(ct_bytes)
                pt = unpad(plain_padded, BLOCK)
                found.append((b, pt))
                break
            elif mode == 'CBC':
                if iv is None:
                    continue
                plain_padded = AES.new(k, AES.MODE_CBC, iv).decrypt(ct_bytes)
                pt = unpad(plain_padded, BLOCK)
                found.append((b, pt))
                break
        except Exception:
            pass
    return found, attempts

# ----------------------------
# Dataset generation
# ----------------------------
SAMPLE_PLAINS = [
    "Attack at dawn. The troops will move at first light.",
    "The quick brown fox jumps over the lazy dog.",
    "Security through obscurity is not a good design.",
    "We will measure success rate and runtime across many samples.",
    "Classical ciphers are educational but not secure by modern standards.",
    "Natural language redundancy helps frequency-based attacks succeed."
]

@dataclass
class Record:
    cipher: str
    plain: str
    cipher_text: str
    true_key: Any
    length: int

def make_classical_records(n_each:int=6) -> List[Record]:
    recs=[]
    # caesar
    for _ in range(n_each):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        k = random.randint(1,25)
        ct = caesar_encrypt(p, k)
        recs.append(Record('caesar', p, ct, k, len(p)))
    # affine
    a_choices = [a for a in range(1,26) if math.gcd(a,26)==1]
    for _ in range(n_each):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        a = random.choice(a_choices)
        b = random.randint(0,25)
        ct = affine_encrypt(p, a, b)
        recs.append(Record('affine', p, ct, (a,b), len(p)))
    # vigenere
    for _ in range(n_each):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        key_len = random.randint(3,6)
        key = ''.join(random.choice(ALPHABET) for _ in range(key_len))
        ct = vigenere_encrypt(p, key)
        recs.append(Record('vigenere', p, ct, key, len(p)))
    return recs

def make_aes_records(n:int=12, modes:Tuple[str,...]=('ECB','CBC','CTR')):
    recs=[]
    for _ in range(n):
        p = random.choice(SAMPLE_PLAINS).encode('utf-8')
        mode = random.choice(modes)
        # generate key
        key = get_random_bytes(16)
        # sometimes simulate all-zero IV for vulnerability
        if mode == 'ECB':
            ct = aes_ecb_encrypt_bytes(p, key)
            true = {'mode':'ECB','key':key.hex(),'iv': None}
        elif mode == 'CBC':
            iv = bytes(16) if random.random() < 0.15 else get_random_bytes(16)
            ct = aes_cbc_encrypt_bytes(p, key, iv)
            true = {'mode':'CBC','key':key.hex(), 'iv': iv.hex() if iv is not None else None}
        elif mode == 'CTR':
            nonce = get_random_bytes(8)
            ct = aes_ctr_encrypt_bytes(p, key, nonce)
            true = {'mode':'CTR','key':key.hex(),'nonce':nonce.hex()}
        else:
            continue
        recs.append(Record('aes', p.decode('utf-8', errors='ignore'), ct.hex(), true, len(p)))
    return recs

# ----------------------------
# CodeLlama integration (classification only)
# ----------------------------
LLAMA_OBJ = None

def init_llama(model_path: str = MODEL_PATH, n_threads:int = LLAMA_THREADS):
    global LLAMA_OBJ
    if not LLAMA_AVAILABLE:
        print("[WARN] llama_cpp not installed — classifier will be simulated.")
        return None
    try:
        LLAMA_OBJ = Llama(model_path=model_path, n_threads=n_threads)
        print(f"[INFO] Loaded Llama model from {model_path}")
        return LLAMA_OBJ
    except Exception as e:
        print("[WARN] Failed to load Llama model:", e)
        LLAMA_OBJ = None
        return None

def parse_llm_json_like(text: str) -> Dict[str,Any]:
    """
    LLMs may not emit strict JSON. We try a forgiving parse:
    - try json.loads first
    - if fails, extract lines 'Mode:' and 'Vulnerabilities:' etc.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # fallback simple parsing
        out = {'mode_guess': None, 'vulnerabilities': [], 'reasoning': text}
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith('mode:'):
                out['mode_guess'] = line.split(':',1)[1].strip().upper()
            elif line.lower().startswith('vulnerabilities:'):
                vals = line.split(':',1)[1].strip()
                out['vulnerabilities'] = [v.strip().lower() for v in vals.split(',') if v.strip()]
        return out

def llama_classify_aes(ct_hex: str, iv_hex: Optional[str]=None, nonce_hex: Optional[str]=None) -> Dict[str,Any]:
    """
    Ask the LLM to classify AES mode and vulnerabilities.
    **Important**: prompt explicitly forbids brute-force or key recovery.
    If the model is not available or USE_LLAMA is False, fall back to a simulated heuristic result.
    """
    # Heuristic fallback first (fast)
    def heuristic_result():
        ct_bytes = bytes.fromhex(ct_hex)
        vulns = []
        mode_guess = 'UNKNOWN'
        if detect_ecb_repetition(ct_bytes):
            mode_guess = 'ECB'
            vulns.append('ecb_repetition')
        # zero IV
        if iv_hex:
            try:
                if detect_all_zero_iv(bytes.fromhex(iv_hex)):
                    if mode_guess == 'UNKNOWN':
                        mode_guess = 'CBC'
                    vulns.append('all_zero_iv')
            except Exception:
                pass
        if detect_padding_like_signature(ct_bytes):
            vulns.append('padding_like')
        if not vulns:
            vulns = ['none']
        return {'mode_guess': mode_guess, 'vulnerabilities': vulns, 'reasoning': 'heuristic'}
    # If not using LLM or llama not loaded, return heuristic
    if (not USE_LLAMA) or (not LLAMA_AVAILABLE) or (LLAMA_OBJ is None):
        return heuristic_result()

    # Build a safe prompt
    prompt = f"""
You are an assistant that can analyze ciphertext structure and metadata for AES-mode classification.
DO NOT attempt to recover keys or run brute-force. This is classification only.

Return a JSON object with keys:
- mode_guess: one of "ECB", "CBC", "CTR", or "UNKNOWN"
- vulnerabilities: a list containing any of "ecb_repetition", "all_zero_iv", "padding_like", "nonce_reuse", or "none"
- reasoning: a short explanation.

Ciphertext(hex): {ct_hex}
IV(hex): {iv_hex if iv_hex else 'NONE'}
Nonce(hex): {nonce_hex if nonce_hex else 'NONE'}
"""
    try:
        resp = LLAMA_OBJ.create_completion(prompt=prompt, max_tokens=256, temperature=0.0, top_p=0.95)
        text = resp.get('choices', [{}])[0].get('text','').strip()
        # Try to parse returned text
        parsed = parse_llm_json_like(text)
        # sanitize
        if parsed.get('mode_guess') is None:
            parsed['mode_guess'] = parsed.get('mode', 'UNKNOWN')
        if 'vulnerabilities' not in parsed:
            parsed['vulnerabilities'] = parsed.get('vulns', ['none'])
        return parsed
    except Exception as e:
        print("[WARN] Llama call failed:", e)
        return heuristic_result()

# ----------------------------
# Experiment runner + evaluation & plotting
# ----------------------------
def run_experiments(run_llm: bool = True, toy_bruteforce_demo: bool = False):
    # Build datasets
    classical = make_classical_records(n_each=6)
    aes_recs = make_aes_records(n=18, modes=('ECB','CBC','CTR'))
    ALL = classical + aes_recs
    print(f"[INFO] Total samples: {len(ALL)} (classical {len(classical)}, aes {len(aes_recs)})")

    rows = []
    for r in ALL:
        row = {'cipher': r.cipher, 'plain': r.plain, 'cipher_text': r.cipher_text, 'true_key': r.true_key}
        # Classical solver results
        if r.cipher == 'caesar':
            t0=time.time(); ck, cp, cs = best_caesar(r.cipher_text); t1=time.time()
            row.update({'classical_key': ck, 'classical_plain': cp, 'classical_time': t1-t0, 'classical_plain_exact': cp.strip().upper() == r.plain.strip().upper()})
        elif r.cipher == 'affine':
            t0=time.time(); ck, cp, cs = best_affine(r.cipher_text); t1=time.time()
            row.update({'classical_key': ck, 'classical_plain': cp, 'classical_time': t1-t0, 'classical_plain_exact': cp.strip().upper() == r.plain.strip().upper()})
        elif r.cipher == 'vigenere':
            t0=time.time(); ck, cp, cs = vigenere_attack(r.cipher_text); t1=time.time()
            row.update({'classical_key': ck, 'classical_plain': cp, 'classical_time': t1-t0, 'classical_plain_exact': cp.strip().upper() == r.plain.strip().upper()})
        elif r.cipher == 'aes':
            # classical heuristics
            ct_bytes = bytes.fromhex(r.cipher_text)
            ecb_guess = detect_ecb_repetition(ct_bytes)
            iv_hex = None
            nonce_hex = None
            if isinstance(r.true_key, dict):
                iv_hex = r.true_key.get('iv')
                nonce_hex = r.true_key.get('nonce')
            row.update({'classical_ecb_flag': ecb_guess, 'classical_iv_all_zero': detect_all_zero_iv(bytes.fromhex(iv_hex)) if iv_hex else False})
        # LLM (classification)
        llm_out = None
        if run_llm and r.cipher == 'aes':
            # call LLM classification
            t0=time.time()
            iv_hex = r.true_key.get('iv') if isinstance(r.true_key, dict) else None
            nonce_hex = r.true_key.get('nonce') if isinstance(r.true_key, dict) else None
            llm_out = llama_classify_aes(r.cipher_text, iv_hex=iv_hex, nonce_hex=nonce_hex)
            t1=time.time()
            row.update({'llm_mode_guess': llm_out.get('mode_guess'), 'llm_vulnerabilities': llm_out.get('vulnerabilities'), 'llm_reasoning': llm_out.get('reasoning'), 'llm_time_s': t1-t0})
        else:
            row.update({'llm_mode_guess': None, 'llm_vulnerabilities': None, 'llm_reasoning': None, 'llm_time_s': None})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "results_crypto_llama.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved results to {csv_path}")

    # Evaluation: mode detection accuracy for AES only
    aes_df = df[df['cipher']=='aes'].copy()
    if not aes_df.empty:
        # classical guess -> 'ECB' if repetition else 'CBC/CTR' we'll set to 'CBC' for demo
        aes_df['classical_mode_guess'] = aes_df['classical_ecb_flag'].map(lambda b: 'ECB' if b else 'NON-ECB')
        aes_df['true_mode'] = aes_df['true_key'].map(lambda d: d.get('mode') if isinstance(d, dict) else None)
        classical_acc = (aes_df['classical_mode_guess'] == aes_df['true_mode']).mean()
        print(f"[METRIC] Classical heuristic mode detection accuracy (ECB vs NON-ECB): {classical_acc:.2%}")

        if run_llm:
            # only count LLM predictions that are not UNKNOWN
            mask = aes_df['llm_mode_guess'].notnull()
            if mask.any():
                llm_acc = (aes_df.loc[mask,'llm_mode_guess'].str.upper() == aes_df.loc[mask,'true_mode']).mean()
                print(f"[METRIC] LLM mode detection accuracy (non-UNKNOWN subset): {llm_acc:.2%}")
            else:
                print("[METRIC] LLM produced no non-UNKNOWN outputs to evaluate.")

        # Plot accuracy bars
        # Build small summary for plotting
        summary_rows = []
        for method in ['classical','llm']:
            if method == 'classical':
                acc = classical_acc
            else:
                acc = llm_acc if run_llm and mask.any() else float('nan')
            summary_rows.append({'method': method, 'accuracy': acc})
        summary_df = pd.DataFrame(summary_rows)
        plt.figure(figsize=(6,4))
        sns.barplot(data=summary_df, x='method', y='accuracy')
        plt.ylim(0,1)
        plt.ylabel('Accuracy')
        plt.title('Mode detection accuracy (ECB vs target)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "mode_detection_accuracy.png"), dpi=200)
        plt.close()
        print(f"[INFO] Saved mode detection plot to {os.path.join(OUT_DIR,'mode_detection_accuracy.png')}")

    # Optional toy brute force on one AES record (very small keyspace)
    if toy_bruteforce_demo:
        print("[INFO] Running toy brute-force demo (ONLY tiny keyspace: 256 candidates).")
        for _, row in aes_df.iterrows():
            true = row['true_key']
            mode = true.get('mode') if isinstance(true, dict) else None
            iv_hex = true.get('iv') if isinstance(true, dict) else None
            ct_bytes = bytes.fromhex(row['cipher_text'])
            iv_bytes = bytes.fromhex(iv_hex) if iv_hex else None
            found, attempts = toy_repeated_byte_bruteforce(ct_bytes, iv_bytes, mode, max_candidates=256)
            print(f"Toy brute-force on record mode={mode}: attempts={attempts}, found={len(found)}")
            if found:
                print(" Found candidate(s) (key_byte, plaintext snippet):")
                for bval, pt in found:
                    print(f"  {bval} -> {pt[:60]}")
            else:
                print(" No toy key recovered (expected for true random keys).")
    return df

# ----------------------------
# Entrypoint
# ----------------------------
import math

def main():
    print("=== Crypto Analysis Demo (safe) ===")
    if USE_LLAMA:
        if not LLAMA_AVAILABLE:
            print("[WARN] llama_cpp not available. Using simulated classifier.")
        else:
            init = init_llama if 'init_llama' in globals() else None
            # init_llama defined earlier as init_llama(...) name mismatch fix
            try:
                init_llama(MODEL_PATH, n_threads=LLAMA_THREADS)
            except Exception as e:
                # fallback handled inside init_llama
                pass

    # Run experiments: enable run_llm if you want classification via Llama (or simulated)
    df = run_experiments(run_llm=USE_LLAMA and LLAMA_AVAILABLE and (LLAMA_OBJ is not None), toy_bruteforce_demo=False)

    print("\nSample of results:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
