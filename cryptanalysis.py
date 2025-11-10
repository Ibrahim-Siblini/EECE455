import os, math, time, random, string, json, textwrap
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from llama_cpp import Llama

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein as Lev

sns.set_context("notebook")

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Backend 
USE_HF = False           
HF_MODEL = "meta-llama/CodeLlama-7b-hf"  

USE_LLAMA_CPP = True     
GGUF_PATH = "codellama-7b.Q5_K_M.gguf"  

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0         
TOP_P = 0.95


OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Backend:", "HF" if USE_HF else ("llama.cpp" if USE_LLAMA_CPP else "None (classical only)"))

# Ciphers

ALPHABET = string.ascii_uppercase


def normalize_text(s: str) -> str:
    s = ''.join(ch for ch in s.upper() if ch.isalpha() or ch.isspace())
    s = ' '.join(s.split())
    return s

# Caesar 

def caesar_encrypt(plain: str, k: int) -> str:
    out = []
    for ch in plain:
        if ch.isalpha():
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
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
    out = []
    for ch in plain:
        if ch.isalpha():
            x = ord(ch) - 65
            out.append(chr((a * x + b) % 26 + 65))
        else:
            out.append(ch)
    return ''.join(out)


def affine_decrypt(ct: str, a: int, b: int) -> str:
    a_inv = _inv_mod_26(a)
    if a_inv is None:
        raise ValueError("No modular inverse for 'a' under mod 26")
    out = []
    for ch in ct:
        if ch.isalpha():
            y = ord(ch) - 65
            x = (a_inv * (y - b)) % 26
            out.append(chr(x + 65))
        else:
            out.append(ch)
    return ''.join(out)


# Vigenère

def vigenere_encrypt(plain: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())
    out = []
    ki = 0
    for ch in plain:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)


def vigenere_decrypt(ct: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())
    out, ki = [], 0
    for ch in ct:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65
            out.append(chr((ord(ch) - 65 - k) % 26 + 65))
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)


# 2) CLASSICAL ATTACKS

ENGLISH_FREQ = {
    'A':0.08167,'B':0.01492,'C':0.02782,'D':0.04253,'E':0.12702,
    'F':0.02228,'G':0.02015,'H':0.06094,'I':0.06966,'J':0.00153,
    'K':0.00772,'L':0.04025,'M':0.02406,'N':0.06749,'O':0.07507,
    'P':0.01929,'Q':0.00095,'R':0.05987,'S':0.06327,'T':0.09056,
    'U':0.02758,'V':0.00978,'W':0.02360,'X':0.00150,'Y':0.01974,'Z':0.00074
}


def english_score(text: str) -> float:
    t = ''.join(ch for ch in text.upper() if ch.isalpha())
    N = len(t)
    if N == 0:
        return -9999.0
    freqs = Counter(t)
    score = 0.0
    for ch, f in ENGLISH_FREQ.items():
        observed = freqs.get(ch, 0) / N
        score -= (observed - f) ** 2
    return score


# Caesar brute force 

def best_caesar(ct: str) -> Tuple[int, str, float]:
    best_s, best_k, best_p = -1e9, None, None
    for k in range(26):
        p = caesar_decrypt(ct, k)
        s = english_score(p)
        if s > best_s:
            best_s, best_k, best_p = s, k, p
    return best_k, best_p, best_s


# Affine brute force 

def best_affine(ct: str) -> Tuple[Tuple[int,int], str, float]:
    best_s, best_key, best_p = -1e9, None, None
    for a in range(1, 26):
        if math.gcd(a, 26) != 1:
            continue
        for b in range(26):
            p = affine_decrypt(ct, a, b)
            s = english_score(p)
            if s > best_s:
                best_s, best_key, best_p = s, (a, b), p
    return best_key, best_p, best_s


# Vigenère (Friedman + per-column Caesar) 

def index_of_coincidence(text: str) -> float:
    t = ''.join(ch for ch in text if ch.isalpha())
    N = len(t)
    if N <= 1:
        return 0.0
    freqs = Counter(t)
    return sum(v * (v - 1) for v in freqs.values()) / (N * (N - 1))


def friedman_estimate(ct: str) -> int:
    ic = index_of_coincidence(ct)
    if ic <= 0:
        return 1
    k = (0.0265 * len(ct)) / ((0.065 - ic) + (len(ct) * (ic - 0.0385)))
    return max(1, int(round(k)))


def vigenere_attack(ct: str, max_len: int = 10) -> Tuple[str, str, float]:
    guess_len = min(max_len, max(1, friedman_estimate(ct)))
    key = ''
    for i in range(guess_len):
        subseq = ''.join(ch for idx, ch in enumerate(ct) if ch.isalpha() and (idx % guess_len) == i)
        best_shift, best_score = 0, -1e9
        for s in range(26):
            dec = ''.join(chr((ord(ch) - 65 - s) % 26 + 65) for ch in subseq)
            sc = english_score(dec)
            if sc > best_score:
                best_shift, best_score = s, sc
        key += chr(best_shift + 65)
    plain = vigenere_decrypt(ct, key)
    return key, plain, english_score(plain)


# 3) LLM INFERENCE (CodeLlama)

# We provide two backends. Set USE_HF or USE_LLAMA_CPP above.

HF_PIPELINE = None
LLAMA_CPP_MODEL = Llama(
    model_path=GGUF_PATH,
    n_threads=12,     # all CPU threads
    n_batch=512       # process up to 512 batches at once
)

if USE_HF:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        _tok = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=False)
        _model = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map="auto", torch_dtype=torch.float16)
        HF_PIPELINE = pipeline("text-generation", model=_model, tokenizer=_tok, device=0)
        print("Loaded HF model:", HF_MODEL)
    except Exception as e:
        print("[WARN] HF backend failed to load:", e)
        USE_HF = False

if USE_LLAMA_CPP:
    try:
        if not os.path.exists(GGUF_PATH):
            print(f"[WARN] GGUF file not found at {GGUF_PATH}. Set GGUF_PATH to your .gguf model.")
        else:
            LLAMA_CPP_MODEL = Llama(model_path=GGUF_PATH)
            print("Loaded llama.cpp model:", GGUF_PATH)
    except Exception as e:
        print("[WARN] llama.cpp backend failed to load:", e)
        USE_LLAMA_CPP = False


def build_prompt(ciphertext: str, cipher_type: str) -> str:
    if cipher_type.lower() == 'caesar':
        return f"""You are a helpful cryptography assistant. Given the ciphertext encrypted with CAESAR (A=0,B=1,...), return the best key (0-25) on a line 'Key:' and the most likely plaintext on a line 'Plaintext:'. Provide only those two lines.

Ciphertext: {ciphertext}
Key:
Plaintext:
"""
    elif cipher_type.lower() == 'vigenere':
        return f"""You are a helpful cryptography assistant. Given the ciphertext encrypted with VIGENERE (A=0,B=1,...), return the key (uppercase letters) on 'Key:' and the most likely plaintext on 'Plaintext:'. Provide only those two lines.

Ciphertext: {ciphertext}
Key:
Plaintext:
"""
    elif cipher_type.lower() == 'affine':
        return f"""You are a helpful cryptography assistant. Given the ciphertext encrypted with AFFINE over uppercase letters (E(x)=(a*x+b) mod 26), return the key as '(a,b)' on 'Key:' and the most likely plaintext on 'Plaintext:'. Provide only those two lines.

Ciphertext: {ciphertext}
Key:
Plaintext:
"""
    else:
        return f"""You are a helpful cryptography assistant. Ciphertext: {ciphertext}
Return 'Key:' and 'Plaintext:' only."""


def llm_generate(prompt: str) -> Tuple[str, float]:
    if USE_HF and HF_PIPELINE is not None:
        t0 = time.time()
        out = HF_PIPELINE(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=(TEMPERATURE>0.0), temperature=TEMPERATURE, top_p=TOP_P, num_return_sequences=1)
        t1 = time.time()
        text = out[0]['generated_text'][len(prompt):].strip()
        return text, (t1 - t0)
    elif USE_LLAMA_CPP and LLAMA_CPP_MODEL is not None:
        t0 = time.time()
        resp = LLAMA_CPP_MODEL.create_completion(
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P
        )           
        t1 = time.time()
        text = resp.get('choices', [{}])[0].get('text', '').strip()
        return text, (t1 - t0)
    else:
        return "", 0.0


def parse_llm_kv(text: str) -> Tuple[Optional[str], Optional[str]]:
    key, plain = None, None
    for line in text.splitlines():
        L = line.strip()
        if L.upper().startswith('KEY:'):
            key = L.split(':', 1)[1].strip()
        elif L.upper().startswith('PLAINTEXT:'):
            plain = L.split(':', 1)[1].strip()
    return key, plain

# 4) DATASET GENERATION

# SAMPLE_PLAINS GENERATED BY CHAT GPT
SAMPLE_PLAINS = [
    "Attack at dawn. The troops will move at first light.",
    "The quick brown fox jumps over the lazy dog.",
    "Security through obscurity is not a good design.",
    "We will measure success rate and runtime across many samples.",
    "Classical ciphers are educational but not secure by modern standards.",
    "Natural language redundancy helps frequency-based attacks succeed.",
]

@dataclass
class Record:
    cipher: str
    plain: str
    cipher_text: str
    true_key: object
    length: int


def make_caesar_records(n: int = 12) -> List[Record]:
    recs = []
    for _ in range(n):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        k = random.randint(1, 25)
        ct = caesar_encrypt(p, k)
        recs.append(Record('caesar', p, ct, k, len(p)))
    return recs


def make_affine_records(n: int = 12) -> List[Record]:
    recs = []
    a_choices = [a for a in range(1, 26) if math.gcd(a, 26) == 1]
    for _ in range(n):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        a = random.choice(a_choices)
        b = random.randint(0, 25)
        ct = affine_encrypt(p, a, b)
        recs.append(Record('affine', p, ct, (a, b), len(p)))
    return recs


def make_vigenere_records(n: int = 12) -> List[Record]:
    recs = []
    for _ in range(n):
        p = normalize_text(random.choice(SAMPLE_PLAINS))
        key_len = random.randint(3, 6)
        key = ''.join(random.choice(ALPHABET) for _ in range(key_len))
        ct = vigenere_encrypt(p, key)
        recs.append(Record('vigenere', p, ct, key, len(p)))
    return recs

# 5) METRICS & HELPERS

def exact_match(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a.strip().upper() == b.strip().upper()


def nlev(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if len(b) == 0:
        return 1.0 if len(a) else 0.0
    return Lev.distance(a, b) / len(b)

# 6) EXPERIMENT RUNNERS

def run_classical_and_llm(records: List[Record]) -> pd.DataFrame:
    rows = []
    for r in records:
        if r.cipher == 'caesar':
            t0 = time.time(); ck, cp, cs = best_caesar(r.cipher_text); t1 = time.time()
            classical_time = t1 - t0
            prompt = build_prompt(r.cipher_text, 'caesar')
        elif r.cipher == 'affine':
            t0 = time.time(); ck, cp, cs = best_affine(r.cipher_text); t1 = time.time()
            classical_time = t1 - t0
            prompt = build_prompt(r.cipher_text, 'affine')
        elif r.cipher == 'vigenere':
            t0 = time.time(); ck, cp, cs = vigenere_attack(r.cipher_text); t1 = time.time()
            classical_time = t1 - t0
            prompt = build_prompt(r.cipher_text, 'vigenere')
        else:
            raise ValueError("Unknown cipher")

        # LLM inference (may be disabled)
        llm_key = llm_plain = llm_raw = None
        llm_time = None
        if USE_HF or USE_LLAMA_CPP:
            llm_raw, llm_time = llm_generate(prompt)
            lk, lp = parse_llm_kv(llm_raw)
            llm_key, llm_plain = lk, lp

        row = {
            'cipher': r.cipher,
            'true_key': r.true_key,
            'plain': r.plain,
            'cipher_text': r.cipher_text,
            'classical_key': ck,
            'classical_plain': cp,
            'classical_time': classical_time,
            'classical_plain_exact': exact_match(cp, r.plain),
            'classical_key_exact': exact_match(str(ck), str(r.true_key)),
            'llm_key': llm_key,
            'llm_plain': llm_plain,
            'llm_time': llm_time,
            'llm_plain_exact': exact_match(llm_plain, r.plain) if llm_plain is not None else None,
            'llm_key_exact': exact_match(str(llm_key), str(r.true_key)) if llm_key is not None else None,
            'llm_lev': nlev(llm_plain or "", r.plain) if llm_plain is not None else None,
            'llm_raw': llm_raw
        }
        rows.append(row)
    return pd.DataFrame(rows)


# 7) RUN EXPERIMENTS

caesar_recs   = make_caesar_records(12)
affine_recs   = make_affine_records(12)
vigenere_recs = make_vigenere_records(12)

ALL = caesar_recs + affine_recs + vigenere_recs

print(f"Total samples: {len(ALL)} (Caesar {len(caesar_recs)}, Affine {len(affine_recs)}, Vigenère {len(vigenere_recs)})")

df = run_classical_and_llm(ALL)

csv_path = os.path.join(OUT_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print("Saved results ->", csv_path)

# Quick peek
pd.set_option('display.max_colwidth', 120)
df.head(8)


# 8) PLOTTING & SUMMARY STATS

# Classical success rates
classical_plain_rates = df.groupby('cipher')['classical_plain_exact'].mean()
classical_key_rates   = df.groupby('cipher')['classical_key_exact'].mean()

print("Classical plaintext exact rates: ", classical_plain_rates)
print("Classical key exact rates: ", classical_key_rates)

# LLM success (where available)
if (USE_HF or USE_LLAMA_CPP) and df['llm_plain_exact'].notnull().any():
    llm_plain_rates = df.groupby('cipher')['llm_plain_exact'].mean()
    print("LLM plaintext exact rates: ", llm_plain_rates)

    plt.figure(figsize=(8,5))
    plot_df = pd.DataFrame({
        'Classical': classical_plain_rates,
        'LLM': llm_plain_rates
    })
    plot_df.plot(kind='bar')
    plt.ylabel('Plaintext exact success rate')
    plt.title('Classical vs CodeLlama (plain exact match)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'success_rates.png'))
    plt.show()

# Runtime comparison (median per cipher)
df['classical_time_ms'] = df['classical_time'] * 1000
if (USE_HF or USE_LLAMA_CPP) and df['llm_time'].notnull().any():
    df['llm_time_ms'] = df['llm_time'] * 1000
    rt = df.groupby('cipher')[['classical_time_ms','llm_time_ms']].median()
    print("Median runtimes (ms):", rt)
    rt.plot(kind='bar', figsize=(8,5))
    plt.ylabel('Median runtime (ms)')
    plt.title('Runtime: Classical vs CodeLlama')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'runtimes.png'))
    plt.show()

# Levenshtein distance (lower is better)
if (USE_HF or USE_LLAMA_CPP) and df['llm_lev'].notnull().any():
    levg = df.groupby('cipher')['llm_lev'].mean()
    print("Mean normalized Levenshtein (LLM):", levg)
    levg.plot(kind='bar', figsize=(7,4))
    plt.ylabel('Mean normalized Levenshtein')
    plt.title('LLM plaintext distance to ground truth')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'levenshtein.png'))
    plt.show()

# 9) MARKDOWN REPORT GENERATOR

def render_bool(x):
    return "—" if pd.isna(x) else ("✅" if bool(x) else "❌")


def generate_report_markdown(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# LLMs for Code‑Based Cryptanalysis with CodeLlama")
    lines.append("")
    lines.append("**Goal.** Evaluate whether CodeLlama can recover plaintext/keys from simplified ciphers (Caesar, Affine, Vigenère), compared to classical algorithmic attacks.")
    lines.append("")
    lines.append("## Methods")
    lines.append("- **Datasets:** Synthetic plaintexts (English sentences) normalized to A–Z; random keys for each cipher.")
    lines.append("- **Ciphers:** Caesar (k∈[1,25]), Affine (a,b) with gcd(a,26)=1, Vigenère (key length 3–6).")
    lines.append("- **Baselines:** Classical solvers (Caesar/Affine brute force; Vigenère via Friedman key‑length estimate + per‑column Caesar).")
    lines.append("- **LLM:** CodeLlama via HuggingFace or llama.cpp; prompts request **only** 'Key:' and 'Plaintext:' lines (no chain‑of‑thought). Temperature=0.0.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("- Exact plaintext recovery rate; exact key recovery rate; normalized Levenshtein; per‑sample runtime.")
    lines.append("")
    # Summary tables
    classical_plain_rates = df.groupby('cipher')['classical_plain_exact'].mean()
    classical_key_rates = df.groupby('cipher')['classical_key_exact'].mean()
    lines.append("## Results (Classical)")
    lines.append("| Cipher | Plaintext exact | Key exact |")
    lines.append("|---|---:|---:|")
    for c in ['caesar','affine','vigenere']:
        if c in classical_plain_rates.index:
            lines.append(f"| {c.title()} | {classical_plain_rates[c]:.2f} | {classical_key_rates[c]:.2f} |")
    if (USE_HF or USE_LLAMA_CPP) and df['llm_plain_exact'].notnull().any():
        llm_plain_rates = df.groupby('cipher')['llm_plain_exact'].mean()
        try:
            llm_key_rates = df.groupby('cipher')['llm_key_exact'].mean()
        except Exception:
            llm_key_rates = None
        lines.append("")
        lines.append("## Results (CodeLlama)")
        lines.append("| Cipher | Plaintext exact | Key exact |")
        lines.append("|---|---:|---:|")
        for c in ['caesar','affine','vigenere']:
            pe = llm_plain_rates.get(c, float('nan')) if hasattr(llm_plain_rates,'get') else float('nan')
            ke = llm_key_rates.get(c, float('nan')) if (llm_key_rates is not None and hasattr(llm_key_rates,'get')) else float('nan')
            lines.append(f"| {c.title()} | {pe if pd.notna(pe) else '—'} | {ke if (llm_key_rates is not None and pd.notna(ke)) else '—'} |")
    lines.append("")
    if (USE_HF or USE_LLAMA_CPP) and df['llm_time'].notnull().any():
        rt = df.groupby('cipher')[['classical_time','llm_time']].median()
        lines.append("## Runtime (median seconds)")
        lines.append("| Cipher | Classical | LLM |")
        lines.append("|---|---:|---:|")
        for c in rt.index:
            lines.append(f"| {c.title()} | {rt.loc[c,'classical_time']:.4f} | {rt.loc[c,'llm_time']:.4f} |")
    lines.append("")
    lines.append("## Discussion")
    lines.append("- Classical solvers are deterministic and near‑instant for these ciphers; they typically achieve near‑perfect recovery on Caesar/Affine and strong performance on Vigenère with adequate text length.")
    lines.append("- CodeLlama can output plausible plaintext for shorter/structured texts, but its success depends on prompt quality and ciphertext length; it may guess fluent English that is **not** the exact original plaintext or key.")
    lines.append("- LLM inference cost and latency are higher than classical brute force for these tasks.")
    lines.append("- Larger models and few‑shot prompts can help, but classic methods remain superior for well‑specified classical ciphers.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("LLMs like CodeLlama are best used as **assistive tools** (e.g., proposing candidates or helping analysts) rather than replacements for classical cryptanalysis on traditional ciphers. Classical algorithms remain more reliable and efficient here.")
    lines.append("")
    lines.append("*Artifacts saved in `outputs/` (CSV + figures).*")
    return "".join(lines)


report_md = generate_report_markdown(df)
report_path = os.path.join(OUT_DIR, "report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)

print("Report saved ->", report_path)
print("=== Report Preview ===")
print("".join(report_md.splitlines()[:60]))


