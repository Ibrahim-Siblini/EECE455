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


