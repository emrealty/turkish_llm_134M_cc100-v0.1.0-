# turkish\_llm\_134M\_cc100 (v0.1.0)

**Türkçe küçük-orta ölçekli dil modeli** — 134M parametre, 32k BPE tokenizer, CC100-tr tabanlı veriyle ön-eğitim. Depoda bir Colab/Jupyter defteri ile veri hazırlama, tokenizer eğitimi ve modeli Hugging Face/Ollama’ya aktarma adımları yer alır.

> **Model adı önerileri**
>
> - Repo: `turkish_llm_134M_cc100`
> - HF Model ID: `turkish-llm-134m-cc100`

---

## Özellikler

- **Parametre sayısı:** \~134M
- **Mimari:** LLaMA-benzeri, **d\_model=768**, **n\_layers=12**, **n\_heads=12**, **head\_dim=64**
- **FFN:** SwiGLU, d\_ff ≈ 2048–2304 (konfige göre)
- **Normalizer:** RMSNorm, **pre-norm**
- **Konumlama:** RoPE (önerilen `rope_theta` → uzun bağlam planına göre ayarlanabilir)
- **Tokenizer:** SentencePiece **BPE-32k**, `character_coverage=0.9995`
- **Bağlam penceresi:** 2048+ (eğitim/çıkarımda ayarlanabilir)
- **Ağırlık bağlama:** Embedding ↔ LM head **weight tying (önerilir)**

## Depo içeriği

- dataset\_cc100.ipynb — veri hazırlama, tokenizer eğitimi, hızlı deneyler
- `requirements.txt` — bağımlılıklar
- `README.md`



## Kurulum

```bash
# (Öneri) Sanal ortam
python -m venv .venv && source .venv/bin/activate

pip install -U pip
pip install torch sentencepiece==0.1.99 fasttext==0.9.2 tqdm xxhash numpy<2
# CUDA sürümüne uygun PyTorch kurulumu için:
# https://pytorch.org/get-started/locally/
```

## Veri

- **Kaynak:** CC100-tr (Common Crawl türevi)
- **Temizlik:** Temel normalizasyon, basit boilerplate ve tekrar (dedup) filtresi önerilir.
- **Token sayısı:** \~2.0B token (hedef). 2.5–3.0B token aralığı verimi iyileştirebilir.
- **Uyarı:** Web kaynaklı metinler önyargı ve hatalar içerebilir. Aşağıdaki “Sorumluluk Notu” bölümüne bakın.

## Tokenizer (BPE-32k)

Notebook’ta örnek komut:

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="/path/to/cc100-tr.clean.txt.gz",
    model_prefix="tr_bpe32k",
    vocab_size=32000,
    character_coverage=0.9995,
    model_type="bpe",
    num_threads=8
)
# (Gelecek sürüm notu) --byte_fallback ile nadir karakter kapsamı artırılabilir.
```

## Mimari konfig (örnek HF config)

> Aşağıdaki sözlük **örnek** olup, kendi eğitim/deneyine göre güncelle.

```jsonc
{
  "model_type": "llama",
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 2048,      // 2048–2304 arası
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,          // uzun bağlam planına göre ayarla
  "vocab_size": 32000,
  "max_position_embeddings": 2048,
  "tie_word_embeddings": true,
  "initializer_range": 0.02
}
```

## Eğitimi çalıştırma (özet)

1. **Veriyi hazırla:** CC100-tr metnini temizle, dedup uygula, tek bir txt.gz halinde tut.
2. **Tokenizer eğit:** Yukarıdaki SPM komutuyla BPE32k üret.
3. **Modeli başlat:** HF `AutoModelForCausalLM.from_config` ile yukarıdaki konfigten başlat.
4. **Ön-eğitim:** Causal LM kaybı (CrossEntropy), AdamW, cosine decay; `sequence_length=2048`, mixed precision (bf16/fp16).
5. **Checkpoint & kaydetme:** `save_pretrained(..., safe_serialization=True)` ile `safetensors` oluştur.

> Colab A100 (80GB) ile 24 saatlik deneme ayarlarına uygun; daha uzun eğitim için adım/epok sayısını artır.

## PyTorch `.pt` → Hugging Face klasörü

`.pt` dosyan yalnızca `state_dict` ise aşağıdaki gibi HF klasörü üret:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

cfg = AutoConfig.from_pretrained("./config_dir_or_dict")
model = AutoModelForCausalLM.from_config(cfg)
state = torch.load("bes_model.pt", map_location="cpu")
model.load_state_dict(state, strict=False)
model.save_pretrained("./turkish_llm_134M_cc100", safe_serialization=True)

# Tokenizer dosyalarını da aynı klasöre kopyala/aktar
# tokenizer = AutoTokenizer.from_pretrained("./tr_bpe32k")
# tokenizer.save_pretrained("./turkish_llm_134M_cc100")
```

## Hızlı çıkarım (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "./turkish_llm_134M_cc100"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

prompt = "Türkiye'nin başkenti neresidir?"
inputs = tok(prompt, return_tensors="pt")
out = mdl.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tok.eos_token_id
)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Ollama ile çalıştırma (GGUF **olmayan** yol — önerilen hızlı yol)

1. HF klasöründe `config.json`, `tokenizer.model/json`, `model.safetensors` hazır olsun.
2. Aynı klasöre `Modelfile` ekle:

```
FROM .
PARAMETER num_ctx 4096
SYSTEM "Türkçe yardımcı asistan."
```

3. Modeli oluştur/çalıştır:

```bash
ollama create tr-134m -f Modelfile
ollama run tr-134m
```

> **Not:** Oluştururken `--quantize q4_K_M` / `q5_K_M` ile daha küçük boyut elde edebilirsin.

## Sorumluluk & sınırlamalar

- Model web türevi verilerle eğitilmiştir; **önyargı, yanlış bilgi ve uygunsuz içerik** üretebilir.
- Medikal/hukuki/finansal kritik alanlarda **uzman doğrulaması** olmadan kullanılmamalıdır.
- Eğitim verilerinin telif/etik durumları kaynak sitelere göre değişebilir; bu depoda **ham veri paylaşımı yoktur**.

## Lisans

- **Kod**: MIT veya Apache-2.0 (tercihine göre bu dosyayı güncelle).
- **Ağırlıklar**: Apache-2.0 + sorumlu kullanım notu **veya** OpenRAIL-M benzeri bir lisans önerilir.

## Yeniden üretilebilirlik

- **Seed:** `42` (örnek)
- **Ortam:** Colab A100 / CUDA 12.x
- **Python:** 3.10+
- **Not:** Notebook’ta “Clear All Outputs” yapıp, Colab’a özgü (Drive mount vb.) satırları `try/except` ile koşullu çalıştırman önerilir.

## Teşekkür

- Common Crawl türev veri iş akışları, CCNet ekosistemi
- Açık kaynak topluluğuna katkı veren tüm proje ve geliştiriciler

---

