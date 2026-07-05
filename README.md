#  Medical Fake News Detection

Vergleich von Large Language Models zur semantischen Fake-News-Erkennung in medizinischen Textdaten.

Dieses Repository enthält den Quellcode zur Bachelorarbeit von Pascal Hartmann
---

##  Projektübersicht

Dieses Projekt implementiert und vergleicht verschiedene Modelle zur automatischen Erkennung von Fake News im medizinischen Bereich. Ziel ist der experimentelle Vergleich klassischer Machine-Learning-Methoden mit Transformer-basierten Sprachmodellen auf dem PubHealth-Datensatz.

---

##  Verwendete Modelle

| Modell | Methode |
|---|---|
| BERT | Full Fine-Tuning |
| RoBERTa | Full Fine-Tuning |
| LLaMA 3.2 3B | LoRA Fine-Tuning (PEFT) |
| TF-IDF + Logistische Regression | Baseline |

---

##  Ergebnisse

| Modell | Beste Strategie | F1-Macro |
|---|---|---|
|  LLaMA 3.2 (LoRA) | title_and_content | 89.14 % |
|  RoBERTa | title_and_content | 86.58 % |
|  BERT | title_and_content | 85.92 % |
| TF-IDF Baseline | content_only | 80.69 % |

---

##  Trainingssetup

- **Datensatz:** PubHealth
- **Split:** 80/20 (stratifiziert, `random_state=42`)
- **Maximale Sequenzlänge:** 512 Tokens
- **Textkombinationsstrategien:** `title_only` · `content_only` · `title_and_content`

---

##  Evaluierungsmetriken

- Accuracy
- Precision & Recall
- F1-Macro-Score
- Konfusionsmatrix

---

##  Reproduzierbarkeit

Zur vollständigen Reproduzierbarkeit wurden folgende Einstellungen fixiert:

- `random_state = 42`
- Identischer Datensatz-Split für alle Modelle
- Einheitliche maximale Tokenlänge (512)
- Gleiche Evaluierungsmetriken für alle Modelle
- Datensatz von (https://github.com/neemakot/Health-Fact-Checking) mit dem Download Link in der Readme unter "DATA" heruntergeladen
---

##  Hardware

- **GPU:** NVIDIA RTX 5080 (16 GB VRAM)
- **Trainings-Präzision:** bf16
- **Hinweis:** LLaMA wurde aufgrund von VRAM-Beschränkungen mit `batch_size=4` trainiert

---

##  Installation

```bash
pip install -r requirements.txt
##  Lizenzen

| Komponente | Lizenz | Quelle |
|---|---|---|
| PubHealth Datensatz | MIT License | [neemakot/Health-Fact-Checking](https://github.com/neemakot/Health-Fact-Checking) |
| BERT | Apache License 2.0 | [google-bert/bert](https://github.com/google-research/bert/blob/master/LICENSE) |
| RoBERTa | MIT License | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/blob/main/LICENSE) |
| LLaMA 3.2 | Meta LLaMA 3.2 Community License | [Meta AI](https://www.llama.com/llama3_2_community_license_agreement) |

> ⚠️ Die Nutzung von LLaMA 3.2 erfordert die Akzeptanz der Meta LLaMA 3.2 Community License auf [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B).


