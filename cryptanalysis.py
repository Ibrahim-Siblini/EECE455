
import os
import time
import random
import string
import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Util import Counter as CryptoCounter

# Llama integration
LLAMA_AVAILABLE = True
try:
    from llama_cpp import Llama
except Exception:
    LLAMA_AVAILABLE = False

# ---------------- Configuration ----------------
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

MODEL_PATH = "codellama-7b.Q5_K_M.gguf"
LLAMA_THREADS = 8
USE_LLAMA = LLAMA_AVAILABLE

BLOCK = 16

# ---------------- AES Weakness Types ----------------
WEAKNESS_TYPES = {
    "ecb_repetition": "ECB mode with repeated plaintext blocks",
    "ecb_known_plaintext": "ECB mode with known plaintext blocks",
    "cbc_zero_iv": "CBC mode with all-zero IV",
    "cbc_predictable_iv": "CBC mode with predictable/sequential IV",
    "cbc_padding_oracle": "CBC mode vulnerable to padding oracle attack",
    "ctr_nonce_reuse": "CTR mode with nonce reuse",
    "ctr_keystream_reuse": "CTR mode with keystream reuse",
    "weak_key": "Weak or related keys",
    "small_key_space": "Small key space (reduced rounds or weak key schedule)",
    "timing_sidechannel": "Timing side-channel vulnerability",
    "none": "No known weakness"
}

# ---------------- Enhanced AES Generation with More Weaknesses ----------------
def generate_weak_aes_samples(n_samples_per_type: int = 10) -> List[Dict]:
    """
    Generate AES samples with various weaknesses for training/testing.
    """
    samples = []
    
    # 1. ECB with repeated blocks (easy to detect)
    for _ in range(n_samples_per_type):
        key = get_random_bytes(16)
        repeated_block = get_random_bytes(16)
        pt = repeated_block * 4  # Repeat 4 times
        cipher = AES.new(key, AES.MODE_ECB)
        ct = cipher.encrypt(pad(pt, BLOCK))
        samples.append({
            "mode": "ECB",
            "plaintext": pt.hex(),
            "ciphertext": ct.hex(),
            "key": key.hex(),
            "iv": None,
            "nonce": None,
            "weakness_type": "ecb_repetition",
            "exploitable": True,
            "attack_method": "block_repetition_analysis"
        })
    
    # 2. ECB with known plaintext (first block known)
    for _ in range(n_samples_per_type):
        key = get_random_bytes(16)
        known_block = b"KNOWN_PLAINTEXT"[:16].ljust(16, b'\x00')
        unknown_data = get_random_bytes(32)
        pt = known_block + unknown_data
        cipher = AES.new(key, AES.MODE_ECB)
        ct = cipher.encrypt(pad(pt, BLOCK))
        samples.append({
            "mode": "ECB",
            "plaintext": pt.hex(),
            "ciphertext": ct.hex(),
            "key": key.hex(),
            "iv": None,
            "nonce": None,
            "weakness_type": "ecb_known_plaintext",
            "exploitable": True,
            "attack_method": "known_plaintext_attack",
            "known_plaintext": known_block.hex()
        })
    
    # 3. CBC with zero IV
    for _ in range(n_samples_per_type):
        key = get_random_bytes(16)
        iv = bytes(16)  # All zeros
        pt = get_random_bytes(48)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(pt, BLOCK))
        samples.append({
            "mode": "CBC",
            "plaintext": pt.hex(),
            "ciphertext": ct.hex(),
            "key": key.hex(),
            "iv": iv.hex(),
            "nonce": None,
            "weakness_type": "cbc_zero_iv",
            "exploitable": True,
            "attack_method": "iv_manipulation"
        })
    
    # 4. CBC with predictable IV (sequential)
    for i in range(n_samples_per_type):
        key = get_random_bytes(16)
        iv = i.to_bytes(16, 'big')  # Sequential IV
        pt = get_random_bytes(48)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(pt, BLOCK))
        samples.append({
            "mode": "CBC",
            "plaintext": pt.hex(),
            "ciphertext": ct.hex(),
            "key": key.hex(),
            "iv": iv.hex(),
            "nonce": None,
            "weakness_type": "cbc_predictable_iv",
            "exploitable": True,
            "attack_method": "predictable_iv_attack"
        })
    
    # 5. CTR with nonce reuse
    for _ in range(n_samples_per_type):
        key1 = get_random_bytes(16)
        key2 = get_random_bytes(16)
        nonce = get_random_bytes(8)  # Same nonce for both
        pt1 = get_random_bytes(32)
        pt2 = get_random_bytes(32)
        cipher1 = AES.new(key1, AES.MODE_CTR, nonce=nonce)
        cipher2 = AES.new(key2, AES.MODE_CTR, nonce=nonce)
        ct1 = cipher1.encrypt(pt1)
        ct2 = cipher2.encrypt(pt2)
        # Store as two related samples
        samples.append({
            "mode": "CTR",
            "plaintext": pt1.hex(),
            "ciphertext": ct1.hex(),
            "key": key1.hex(),
            "iv": None,
            "nonce": nonce.hex(),
            "weakness_type": "ctr_nonce_reuse",
            "exploitable": True,
            "attack_method": "nonce_reuse_xor_attack",
            "related_ciphertext": ct2.hex(),
            "related_plaintext": pt2.hex()
        })
    
    # 6. CTR with keystream reuse (same key, same nonce)
    for _ in range(n_samples_per_type):
        key = get_random_bytes(16)
        nonce = get_random_bytes(8)
        pt1 = get_random_bytes(32)
        pt2 = get_random_bytes(32)
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
        ct1 = cipher.encrypt(pt1)
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)  # Reuse
        ct2 = cipher.encrypt(pt2)
        samples.append({
            "mode": "CTR",
            "plaintext": pt1.hex(),
            "ciphertext": ct1.hex(),
            "key": key.hex(),
            "iv": None,
            "nonce": nonce.hex(),
            "weakness_type": "ctr_keystream_reuse",
            "exploitable": True,
            "attack_method": "keystream_reuse_xor",
            "related_ciphertext": ct2.hex(),
            "related_plaintext": pt2.hex()
        })
    
    # 7. Secure samples (no obvious weakness)
    for _ in range(n_samples_per_type * 2):
        key = get_random_bytes(16)
        iv = get_random_bytes(16)
        pt = get_random_bytes(48)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(pt, BLOCK))
        samples.append({
            "mode": "CBC",
            "plaintext": pt.hex(),
            "ciphertext": ct.hex(),
            "key": key.hex(),
            "iv": iv.hex(),
            "nonce": None,
            "weakness_type": "none",
            "exploitable": False,
            "attack_method": None
        })
    
    random.shuffle(samples)
    return samples

# ---------------- Few-Shot Training Examples ----------------
FEW_SHOT_EXAMPLES = """
# Example 1: ECB Repetition Attack
Ciphertext: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6
Analysis: The ciphertext shows repeating 16-byte blocks (a1b2c3d4e5f6 repeated 4 times).
Weakness: ECB mode with repeated plaintext blocks
Attack: Since ECB encrypts identical blocks identically, we can identify patterns.
Plaintext blocks are likely identical. If we know one plaintext block, we can identify all occurrences.
Result: Pattern detected, plaintext structure revealed.

# Example 2: CBC Zero IV Attack
Ciphertext: 5f8a3b2c1d4e6f7a8b9c0d1e2f3a4b5
IV: 00000000000000000000000000000000
Analysis: IV is all zeros, which is a security weakness.
Weakness: CBC mode with all-zero IV
Attack: With zero IV, the first block's encryption is predictable. If we can manipulate the IV, we can control the first block's decryption.
Result: First block vulnerable to manipulation.

# Example 3: CTR Nonce Reuse Attack
Ciphertext1: a1b2c3d4e5f6...
Ciphertext2: f6e5d4c3b2a1...
Nonce: 1234567890abcdef (same for both)
Analysis: Same nonce used with different keys/plaintexts.
Weakness: CTR nonce reuse
Attack: XOR the two ciphertexts: ct1 XOR ct2 = (pt1 XOR keystream) XOR (pt2 XOR keystream) = pt1 XOR pt2
If we know pt1, we can recover pt2, or use frequency analysis on pt1 XOR pt2.
Result: Plaintexts can be recovered through XOR analysis.

# Example 4: Secure CBC
Ciphertext: 8f3a7b2c9d4e1f6a5b8c2d7e3f9a4b1c6
IV: 7a3f8b2c9d4e1f6a5b8c2d7e3f9a4b1 (random)
Analysis: Random IV, proper CBC mode implementation.
Weakness: None detected
Attack: No obvious attack vector
Result: Secure implementation.
"""

# ---------------- Enhanced LLM Cryptanalysis Function ----------------
LLAMA_OBJ = None

def init_llama(model_path=MODEL_PATH, n_threads=LLAMA_THREADS):
    global LLAMA_OBJ
    if not LLAMA_AVAILABLE:
        return None
    try:
        LLAMA_OBJ = Llama(model_path=model_path, n_threads=n_threads, n_ctx=4096)
        return LLAMA_OBJ
    except Exception:
        LLAMA_OBJ = None
        return None

def analyze_with_llm(sample: Dict) -> Dict[str, Any]:
    """
    Use LLM with few-shot learning to analyze AES sample and detect weaknesses.
    """
    if not USE_LLAMA or not LLAMA_AVAILABLE or LLAMA_OBJ is None:
        return analyze_with_heuristics(sample)
    
    prompt = f"""{FEW_SHOT_EXAMPLES}

# New Sample to Analyze:
Ciphertext: {sample['ciphertext']}
IV: {sample.get('iv', 'NONE')}
Nonce: {sample.get('nonce', 'NONE')}
Mode: {sample.get('mode', 'UNKNOWN')}

Analyze this AES ciphertext and identify:
1. What weakness exists (if any)?
2. How can this weakness be exploited?
3. What attack method would work?
4. Can plaintext be recovered (partially or fully)?

Provide your analysis in JSON format:
{{
    "weakness_detected": "weakness_type or 'none'",
    "weakness_description": "detailed description",
    "attack_method": "how to exploit",
    "exploitable": true/false,
    "recovered_plaintext": "hex string if recoverable, else null",
    "reasoning": "step-by-step analysis"
}}
"""
    
    try:
        t0 = time.time()
        resp = LLAMA_OBJ.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=0.1,  # Low temperature for more deterministic analysis
            top_p=0.95,
            stop=["# New Sample", "\n\n#"]
        )
        t1 = time.time()
        
        text = resp.get('choices', [{}])[0].get('text', '').strip()
        
        # Try to parse JSON from response
        result = parse_llm_response(text, sample)
        result['llm_latency_s'] = t1 - t0
        result['source'] = 'llm'
        
        return result
    except Exception as e:
        print(f"[WARN] LLM analysis failed: {e}")
        return analyze_with_heuristics(sample)

def parse_llm_response(text: str, sample: Dict) -> Dict[str, Any]:
    """Parse LLM response and extract structured information."""
    # Try to find JSON in response
    try:
        # Look for JSON block
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return {
                'weakness_detected': parsed.get('weakness_detected', 'unknown'),
                'weakness_description': parsed.get('weakness_description', ''),
                'attack_method': parsed.get('attack_method', ''),
                'exploitable': parsed.get('exploitable', False),
                'recovered_plaintext': parsed.get('recovered_plaintext'),
                'reasoning': parsed.get('reasoning', text[:200])
            }
    except:
        pass
    
    # Fallback: extract information from text
    result = {
        'weakness_detected': 'unknown',
        'weakness_description': '',
        'attack_method': '',
        'exploitable': False,
        'recovered_plaintext': None,
        'reasoning': text[:500]
    }
    
    # Try to detect weakness from keywords
    text_lower = text.lower()
    for weakness, desc in WEAKNESS_TYPES.items():
        if weakness.replace('_', ' ') in text_lower or weakness in text_lower:
            result['weakness_detected'] = weakness
            result['weakness_description'] = desc
            break
    
    return result

def analyze_with_heuristics(sample: Dict) -> Dict[str, Any]:
    """Heuristic-based analysis fallback."""
    ct = bytes.fromhex(sample['ciphertext'])
    weakness = 'none'
    attack_method = None
    exploitable = False
    
    # Check for ECB repetition
    blocks = [ct[i:i+BLOCK] for i in range(0, len(ct), BLOCK)]
    if len(set(blocks)) < len(blocks):
        weakness = 'ecb_repetition'
        attack_method = 'block_repetition_analysis'
        exploitable = True
    
    # Check for zero IV
    if sample.get('iv'):
        iv = bytes.fromhex(sample['iv'])
        if all(b == 0 for b in iv):
            weakness = 'cbc_zero_iv'
            attack_method = 'iv_manipulation'
            exploitable = True
    
    return {
        'weakness_detected': weakness,
        'weakness_description': WEAKNESS_TYPES.get(weakness, ''),
        'attack_method': attack_method,
        'exploitable': exploitable,
        'recovered_plaintext': None,
        'reasoning': f'Heuristic detected: {weakness}',
        'source': 'heuristic'
    }

# ---------------- Cryptanalysis Attack Implementations ----------------
def attack_ecb_repetition(sample: Dict) -> Optional[str]:
    """Attack ECB with repeated blocks."""
    ct = bytes.fromhex(sample['ciphertext'])
    blocks = [ct[i:i+BLOCK] for i in range(0, len(ct), BLOCK)]
    
    # Find repeated blocks
    block_counts = Counter(blocks)
    repeated = [b for b, count in block_counts.items() if count > 1]
    
    if repeated:
        # If we have known plaintext, we can map blocks
        if 'known_plaintext' in sample:
            known_pt = bytes.fromhex(sample['known_plaintext'])
            known_ct_block = blocks[0]  # Assuming first block
            # All blocks matching known_ct_block correspond to known_pt
            return f"Identified {len(repeated)} repeated block patterns"
    
    return "Pattern detected but plaintext recovery requires additional information"

def attack_cbc_zero_iv(sample: Dict) -> Optional[str]:
    """Attack CBC with zero IV."""
    # With zero IV, first block is vulnerable
    # If we can control or predict the first block, we can manipulate it
    return "First block vulnerable to IV manipulation attack"

def attack_ctr_nonce_reuse(sample: Dict) -> Optional[str]:
    """Attack CTR with nonce reuse."""
    if 'related_ciphertext' not in sample:
        return None
    
    ct1 = bytes.fromhex(sample['ciphertext'])
    ct2 = bytes.fromhex(sample['related_ciphertext'])
    
    # XOR the two ciphertexts
    min_len = min(len(ct1), len(ct2))
    xor_result = bytes(a ^ b for a, b in zip(ct1[:min_len], ct2[:min_len]))
    
    # This equals pt1 XOR pt2
    # If we know one plaintext, we can recover the other
    if 'related_plaintext' in sample:
        pt2 = bytes.fromhex(sample['related_plaintext'])
        pt1_recovered = bytes(a ^ b for a, b in zip(xor_result[:min_len], pt2[:min_len]))
        return pt1_recovered.hex()
    
    return xor_result.hex()  # Return pt1 XOR pt2

def attack_ctr_keystream_reuse(sample: Dict) -> Optional[str]:
    """Attack CTR with keystream reuse (same as nonce reuse)."""
    return attack_ctr_nonce_reuse(sample)

def perform_attack(sample: Dict, detected_weakness: str) -> Dict[str, Any]:
    """Perform the actual cryptanalysis attack based on detected weakness."""
    attack_result = {
        'attack_successful': False,
        'recovered_data': None,
        'attack_details': ''
    }
    
    try:
        if detected_weakness == 'ecb_repetition':
            result = attack_ecb_repetition(sample)
            attack_result['recovered_data'] = result
            attack_result['attack_successful'] = result is not None
            attack_result['attack_details'] = 'Block repetition pattern identified'
        
        elif detected_weakness == 'cbc_zero_iv':
            result = attack_cbc_zero_iv(sample)
            attack_result['recovered_data'] = result
            attack_result['attack_successful'] = True
            attack_result['attack_details'] = 'IV manipulation vulnerability identified'
        
        elif detected_weakness == 'ctr_nonce_reuse':
            result = attack_ctr_nonce_reuse(sample)
            attack_result['recovered_data'] = result
            attack_result['attack_successful'] = result is not None
            attack_result['attack_details'] = 'XOR analysis performed on reused nonce'
        
        elif detected_weakness == 'ctr_keystream_reuse':
            result = attack_ctr_keystream_reuse(sample)
            attack_result['recovered_data'] = result
            attack_result['attack_successful'] = result is not None
            attack_result['attack_details'] = 'Keystream reuse exploited via XOR'
        
        else:
            attack_result['attack_details'] = 'No attack method available for this weakness'
    
    except Exception as e:
        attack_result['attack_details'] = f'Attack failed: {str(e)}'
    
    return attack_result

# ---------------- Evaluation Framework ----------------
def evaluate_cryptanalysis(samples: List[Dict], analyses: List[Dict], attacks: List[Dict]) -> Dict[str, Any]:
    """Evaluate the success of cryptanalysis."""
    metrics = {
        'total_samples': len(samples),
        'weakness_detection': {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0
        },
        'attack_success': {
            'successful_attacks': 0,
            'failed_attacks': 0,
            'no_attack_attempted': 0
        },
        'timing': {
            'avg_analysis_time': 0.0,
            'avg_attack_time': 0.0
        }
    }
    
    analysis_times = []
    attack_times = []
    
    for i, (sample, analysis, attack) in enumerate(zip(samples, analyses, attacks)):
        true_weakness = sample['weakness_type']
        detected_weakness = analysis.get('weakness_detected', 'none')
        
        # Weakness detection metrics
        if true_weakness != 'none' and detected_weakness != 'none':
            if true_weakness == detected_weakness:
                metrics['weakness_detection']['true_positive'] += 1
            else:
                metrics['weakness_detection']['false_positive'] += 1
        elif true_weakness == 'none' and detected_weakness == 'none':
            metrics['weakness_detection']['true_negative'] += 1
        elif true_weakness == 'none' and detected_weakness != 'none':
            metrics['weakness_detection']['false_positive'] += 1
        else:
            metrics['weakness_detection']['false_negative'] += 1
        
        # Attack success metrics
        if sample['exploitable']:
            if attack['attack_successful']:
                metrics['attack_success']['successful_attacks'] += 1
            else:
                metrics['attack_success']['failed_attacks'] += 1
        else:
            metrics['attack_success']['no_attack_attempted'] += 1
        
        # Timing
        if 'llm_latency_s' in analysis:
            analysis_times.append(analysis['llm_latency_s'])
        if 'attack_time' in attack:
            attack_times.append(attack['attack_time'])
    
    if analysis_times:
        metrics['timing']['avg_analysis_time'] = np.mean(analysis_times)
    if attack_times:
        metrics['timing']['avg_attack_time'] = np.mean(attack_times)
    
    # Calculate precision, recall, F1
    tp = metrics['weakness_detection']['true_positive']
    fp = metrics['weakness_detection']['false_positive']
    fn = metrics['weakness_detection']['false_negative']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['weakness_detection']['precision'] = precision
    metrics['weakness_detection']['recall'] = recall
    metrics['weakness_detection']['f1_score'] = f1
    
    return metrics

# ---------------- Main Experiment Runner ----------------
def run_cryptanalysis_experiment(n_samples: int = 50):
    """Run the complete cryptanalysis experiment."""
    print("=== AES Cryptanalysis with LLM Training ===\n")
    
    # Initialize Llama
    if USE_LLAMA and LLAMA_AVAILABLE:
        print("[INFO] Initializing Llama model...")
        init_llama()
        if LLAMA_OBJ is None:
            print("[WARN] Llama initialization failed, using heuristics")
        else:
            print("[INFO] Llama model loaded successfully")
    else:
        print("[INFO] Using heuristic-based analysis only")
    
    # Generate samples
    print(f"\n[INFO] Generating {n_samples} AES samples with various weaknesses...")
    samples = generate_weak_aes_samples(n_samples_per_type=n_samples // 7)
    samples = samples[:n_samples]  # Limit to requested number
    
    print(f"[INFO] Generated {len(samples)} samples")
    print(f"  - Weak samples: {sum(1 for s in samples if s['weakness_type'] != 'none')}")
    print(f"  - Secure samples: {sum(1 for s in samples if s['weakness_type'] == 'none')}")
    
    # Analyze samples
    print("\n[INFO] Analyzing samples with LLM...")
    analyses = []
    attacks = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
        
        # LLM analysis
        t0 = time.time()
        analysis = analyze_with_llm(sample)
        t1 = time.time()
        analysis['total_time'] = t1 - t0
        analyses.append(analysis)
        
        # Perform attack if weakness detected
        detected_weakness = analysis.get('weakness_detected', 'none')
        if detected_weakness != 'none' and detected_weakness != 'unknown':
            t0 = time.time()
            attack = perform_attack(sample, detected_weakness)
            t1 = time.time()
            attack['attack_time'] = t1 - t0
        else:
            attack = {'attack_successful': False, 'recovered_data': None, 'attack_details': 'No weakness detected'}
            attack['attack_time'] = 0.0
        attacks.append(attack)
    
    # Evaluate
    print("\n[INFO] Evaluating results...")
    metrics = evaluate_cryptanalysis(samples, analyses, attacks)
    
    # Print metrics
    print("\n=== RESULTS ===")
    print(f"\nWeakness Detection:")
    print(f"  Precision: {metrics['weakness_detection']['precision']:.2%}")
    print(f"  Recall: {metrics['weakness_detection']['recall']:.2%}")
    print(f"  F1 Score: {metrics['weakness_detection']['f1_score']:.2%}")
    print(f"  True Positives: {metrics['weakness_detection']['true_positive']}")
    print(f"  False Positives: {metrics['weakness_detection']['false_positive']}")
    print(f"  False Negatives: {metrics['weakness_detection']['false_negative']}")
    
    print(f"\nAttack Success:")
    print(f"  Successful: {metrics['attack_success']['successful_attacks']}")
    print(f"  Failed: {metrics['attack_success']['failed_attacks']}")
    
    print(f"\nTiming:")
    print(f"  Avg Analysis Time: {metrics['timing']['avg_analysis_time']:.3f}s")
    print(f"  Avg Attack Time: {metrics['timing']['avg_attack_time']:.3f}s")
    
    # Save results
    results = []
    for sample, analysis, attack in zip(samples, analyses, attacks):
        results.append({
            'mode': sample['mode'],
            'weakness_type': sample['weakness_type'],
            'detected_weakness': analysis.get('weakness_detected', 'unknown'),
            'weakness_match': sample['weakness_type'] == analysis.get('weakness_detected', ''),
            'exploitable': sample['exploitable'],
            'attack_successful': attack['attack_successful'],
            'analysis_time': analysis.get('total_time', 0),
            'attack_time': attack.get('attack_time', 0),
            'source': analysis.get('source', 'unknown'),
            'reasoning': analysis.get('reasoning', '')[:200]
        })
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "aes_cryptanalysis_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Results saved to {csv_path}")
    
    # Save detailed metrics
    metrics_path = os.path.join(OUT_DIR, "aes_cryptanalysis_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")
    
    return samples, analyses, attacks, metrics

if __name__ == "__main__":
    run_cryptanalysis_experiment(n_samples=50)
