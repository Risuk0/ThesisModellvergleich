##Projektübersicht

Dieses Projekt implementiert und vergleicht verschiedene Modelle zur automatischen Erkennung von Fake News im medizinischen Bereich unter Verwendung des PubHealth-Datensatzes. Ziel ist der experimentelle Vergleich von klassischen Machine-Learning-Methoden und Transformer-basierten Sprachmodellen.

Verwendete Modelle

In diesem Projekt wurden folgende Modelle verwendet:

BERT
RoBERTa
LLaMA 3.2 (LoRA Fine-Tuning)
TF-IDF + Logistische Regression (Baseline)
Textkombinationsstrategien

Die Modelle wurden mit folgenden Eingabestrategien getestet:

title_only
content_only
title_and_content
Trainingssetup
Datensatz: PubHealth
Split: 80/20 stratifiziert
Random State: 42
Maximale Sequenzlänge: 512 Tokens
Optimierung:
BERT / RoBERTa: Full Fine-Tuning
LLaMA 3.2: LoRA (parameter-efficient fine-tuning)
Evaluation

Die Modelle wurden mit folgenden Metriken bewertet:

Accuracy
Precision
Recall
F1-Macro-Score
Confusion Matrix
Ergebnisse (Best Model Comparison)
Modell	Beste Strategie	F1-Macro
LLaMA 3.2 (LoRA)	title_and_content	89.14%
RoBERTa	title_and_content	86.58%
BERT	title_and_content	85.92%
TF-IDF Baseline	content_only	80.69%
Reproduzierbarkeit

Zur Reproduzierbarkeit wurden folgende Einstellungen fixiert:

random_state = 42
identischer Datensatz-Split
gleiche Evaluierungsmetriken
feste maximale Tokenlänge (512)
Installation
pip install -r requirements.txt
Nutzung

Hardware
GPU: NVIDIA RTX 5080 (16GB VRAM)
LLaMA Training mit reduzierter Batch Size (4 statt 16)
Lizenz und Drittmodelle

Dieses Projekt verwendet Modelle unter unterschiedlichen Lizenzen:

BERT
Lizenz: Apache License 2.0
Quelle: Original BERT Paper (Devlin et al., 2019)
RoBERTa
Lizenz: Apache License 2.0
Quelle: Liu et al. (2019)
LLaMA 3.2
Lizenz: Meta LLaMA 3.2 Community License
Nutzung unter Einhaltung der Meta-Nutzungsbedingungen
