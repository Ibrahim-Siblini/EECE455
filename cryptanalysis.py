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

sns.set_context("notebook")  # set seaborn plotting context suitable for notebooks

RNG_SEED = 42  # deterministic randomness seed constant
random.seed(RNG_SEED)  # seed python random
np.random.seed(RNG_SEED)  # seed numpy random

# Backend selection flags and model identifiers
USE_HF = False           # whether to use HuggingFace pipeline backend
HF_MODEL = "meta-llama/CodeLlama-7b-hf"  # HF model id (if using HF)

USE_LLAMA_CPP = True     # whether to use local llama.cpp backend
GGUF_PATH = "codellama-7b.Q5_K_M.gguf"  # path to local .gguf quantized model

MAX_NEW_TOKENS = 256  # LLM generation length limit
TEMPERATURE = 0.0     # deterministic generation (0 => greedy)
TOP_P = 0.95          # nucleus sampling cutoff (unused if temp=0)

OUT_DIR = "./outputs"  # directory to save CSV, plots, report
os.makedirs(OUT_DIR, exist_ok=True)  # ensure output dir exists

print("Backend:", "HF" if USE_HF else ("llama.cpp" if USE_LLAMA_CPP else "None (classical only)"))
# print what backend is chosen for clarity

# Ciphers
ALPHABET = string.ascii_uppercase  # "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def normalize_text(s: str) -> str:
    # Normalize text to uppercase letters and single spaces, preserving spaces between words
    s = ''.join(ch for ch in s.upper() if ch.isalpha() or ch.isspace())
    s = ' '.join(s.split())  # collapse multiple whitespace to single spaces
    return s

# Caesar cipher: shift by k (A=0)
def caesar_encrypt(plain: str, k: int) -> str:
    out = []
    for ch in plain:
        if ch.isalpha():  # only shift letters
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))  # shift and wrap in A-Z
        else:
            out.append(ch)  # keep spaces/punctuation unchanged
    return ''.join(out)

def caesar_decrypt(ct: str, k: int) -> str:
    # Decrypt by encrypting with negative shift (mod 26)
    return caesar_encrypt(ct, (-k) % 26)

# Affine cipher helpers and implementation: E(x) = (a*x + b) mod 26
def _inv_mod_26(a: int) -> Optional[int]:
    # Find modular inverse of 'a' modulo 26 (if exists), else None
    for i in range(26):
        if (a * i) % 26 == 1:
            return i
    return None

def affine_encrypt(plain: str, a: int, b: int) -> str:
    out = []
    for ch in plain:
        if ch.isalpha():
            x = ord(ch) - 65  # map A->0 .. Z->25
            out.append(chr((a * x + b) % 26 + 65))  # affine transformation
        else:
            out.append(ch)
    return ''.join(out)

def affine_decrypt(ct: str, a: int, b: int) -> str:
    a_inv = _inv_mod_26(a)  # compute inverse of a mod 26
    if a_inv is None:
        raise ValueError("No modular inverse for 'a' under mod 26")  # invalid 'a' value
    out = []
    for ch in ct:
        if ch.isalpha():
            y = ord(ch) - 65
            x = (a_inv * (y - b)) % 26  # invert affine: x = a^{-1} * (y - b) mod 26
            out.append(chr(x + 65))
        else:
            out.append(ch)
    return ''.join(out)

# Vigenère cipher implementation (repeating key)
def vigenere_encrypt(plain: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())  # sanitize key
    out = []
    ki = 0  # key index
    for ch in plain:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65  # shift amount from key letter
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)

def vigenere_decrypt(ct: str, key: str) -> str:
    key = ''.join(k for k in key.upper() if k.isalpha())  # sanitize key
    out, ki = [], 0
    for ch in ct:
        if ch.isalpha():
            k = ord(key[ki % len(key)]) - 65
            out.append(chr((ord(ch) - 65 - k) % 26 + 65))  # reverse shift
            ki += 1
        else:
            out.append(ch)
    return ''.join(out)

# 2) CLASSICAL ATTACKS - frequency analysis scoring

# English letter frequency table used for scoring plaintext candidates
ENGLISH_FREQ = {
    'A':0.08167,'B':0.01492,'C':0.02782,'D':0.04253,'E':0.12702,
    'F':0.02228,'G':0.02015,'H':0.06094,'I':0.06966,'J':0.00153,
    'K':0.00772,'L':0.04025,'M':0.02406,'N':0.06749,'O':0.07507,
    'P':0.01929,'Q':0.00095,'R':0.05987,'S':0.06327,'T':0.09056,
    'U':0.02758,'V':0.00978,'W':0.02360,'X':0.00150,'Y':0.01974,'Z':0.00074
}

def english_score(text: str) -> float:
    # Score how close the letter distribution of text is to expected English frequencies.
    t = ''.join(ch for ch in text.upper() if ch.isalpha())  # remove non-letters
    N = len(t)
    if N == 0:
        return -9999.0  # very bad score if no letters
    freqs = Counter(t)
    score = 0.0
    for ch, f in ENGLISH_FREQ.items():
        observed = freqs.get(ch, 0) / N  # observed frequency
        score -= (observed - f) ** 2  # negative squared error (lower error => higher score)
    return score

# Caesar brute force by trying all 26 shifts and scoring with english_score
def best_caesar(ct: str) -> Tuple[int, str, float]:
    best_s, best_k, best_p = -1e9, None, None
    for k in range(26):
        p = caesar_decrypt(ct, k)  # try shift k
        s = english_score(p)  # score the candidate plaintext
        if s > best_s:
            best_s, best_k, best_p = s, k, p  # keep best
    return best_k, best_p, best_s  # return key, plaintext, score

# Affine brute force: iterate over all 'a' coprime with 26 and all b in 0..25
def best_affine(ct: str) -> Tuple[Tuple[int,int], str, float]:
    best_s, best_key, best_p = -1e9, None, None
    for a in range(1, 26):
        if math.gcd(a, 26) != 1:
            continue  # skip 'a' values without modular inverse
        for b in range(26):
            p = affine_decrypt(ct, a, b)  # decrypt candidate
            s = english_score(p)
            if s > best_s:
                best_s, best_key, best_p = s, (a, b), p
    return best_key, best_p, best_s  # return (a,b), plaintext, score

# Vigenère attack utilities: index of coincidence and Friedman estimate
def index_of_coincidence(text: str) -> float:
    t = ''.join(ch for ch in text if ch.isalpha())
    N = len(t)
    if N <= 1:
        return 0.0
    freqs = Counter(t)
    # sum_{letters} v*(v-1) / (N*(N-1))
    return sum(v * (v - 1) for v in freqs.values()) / (N * (N - 1))

def friedman_estimate(ct: str) -> int:
    # Friedman formula to estimate key length from index of coincidence
    ic = index_of_coincidence(ct)
    if ic <= 0:
        return 1
    k = (0.0265 * len(ct)) / ((0.065 - ic) + (len(ct) * (ic - 0.0385)))
    return max(1, int(round(k)))  # ensure at least key length 1

def vigenere_attack(ct: str, max_len: int = 10) -> Tuple[str, str, float]:
    # Estimate key length then solve each column as Caesar
    guess_len = min(max_len, max(1, friedman_estimate(ct)))  # choose guessed key length
    key = ''
    for i in range(guess_len):
        # build subsequence of every guess_len-th letter starting at position i
        subseq = ''.join(ch for idx, ch in enumerate(ct) if ch.isalpha() and (idx % guess_len) == i)
        best_shift, best_score = 0, -1e9
        for s in range(26):  # try all Caesar shifts for this column
            dec = ''.join(chr((ord(ch) - 65 - s) % 26 + 65) for ch in subseq)
            sc = english_score(dec)
            if sc > best_score:
                best_shift, best_score = s, sc
        key += chr(best_shift + 65)  # convert shift to uppercase letter
    plain = vigenere_decrypt(ct, key)  # decrypt with reconstructed key
    return key, plain, english_score(plain)

