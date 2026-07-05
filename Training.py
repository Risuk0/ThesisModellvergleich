import pandas as pd
import numpy as np
import torch
from huggingface_hub import login
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

login(token="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")  # deinen Token hier einfügen

# Device Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")



# Load Dataset

def load_dataset_tsv(folder="."):

    print("Load PUBHEALTH from folder")

    train = pd.read_csv(f"{folder}/train.tsv", sep="\t")
    dev   = pd.read_csv(f"{folder}/dev.tsv",   sep="\t")
    test  = pd.read_csv(f"{folder}/test.tsv",  sep="\t")

    df = pd.concat([train, dev, test], ignore_index=True)

    df = df.rename(columns={
        "claim":     "title",
        "main_text": "text"
    })

    df = df.dropna(subset=["label", "title"])

    label_map = {
        "true":     1,
        "false":    0,
        "mixture":  0,
        "unproven": 0
    }
    df = df[df["label"].isin(label_map.keys())]
    df["label"] = df["label"].map(label_map)

    print(f"Datensatz geladen: {len(df)} Einträge")
    print(f"Real (1): {df['label'].sum()} | Fake (0): {(df['label'] == 0).sum()}")
    print(f"Klassenverteilung: Real {df['label'].mean()*100:.1f}% ; Fake {(1-df['label'].mean())*100:.1f}%")

    return df



# Dataset Class

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }



# Text-Combination

def title_content_combination(df, strategy="title_and_content"):

    combined_texts = []

    for _, row in df.iterrows():
        title   = str(row.get("title", "")) if "title" in df.columns else ""
        content = str(row.get("text",  "")) if "text"  in df.columns else ""

        if strategy == "title_only":
            final_text = title

        elif strategy == "content_only":
            final_text = content

        elif strategy == "title_and_content":
            if title and content:
                final_text = f"{title} [SEP] {content}"
            elif title:
                final_text = title
            elif content:
                final_text = content
            else:
                final_text = ""

        else:
            raise ValueError(f"Unbekannte Strategie: {strategy}."
                             f"Options: title_only, content_only, title_and_content")

        combined_texts.append(final_text.strip())

    return combined_texts



# Preprocessing

def preprocess(df, text_strategy="title_and_content"):
    combined_texts = title_content_combination(df, text_strategy)
    labels = df["label"].tolist()

    valid_indices  = [i for i, t in enumerate(combined_texts) if t.strip()]
    combined_texts = [combined_texts[i] for i in valid_indices]
    labels         = [labels[i]         for i in valid_indices]

    print(f"\nTextstrategie: {text_strategy}")
    print(f"Verarbeitete Einträge: {len(labels)}")

    return combined_texts, labels



# Text-length

def analyze_text_lengths(texts, strategy_name=""):
    lengths = [len(t.split()) for t in texts]
    print(f"\nTextlängenanalyse [{strategy_name}]")
    print(f" Durchschnitt : {np.mean(lengths):.1f} Wörter")
    print(f" Median       : {np.median(lengths):.1f} Wörter")
    print(f" Max          : {np.max(lengths)} Wörter")
    print(f" Min          : {np.min(lengths)} Wörter")
    print(f" 95. Perzentil: {np.percentile(lengths, 95):.1f} Wörter")

    truncated = sum(1 for l in lengths if l > 512)
    print(f" Texte > 512 Wörter (Truncation): {truncated} ({truncated/len(lengths)*100:.1f}%)")

    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50, color="steelblue", edgecolor="black")
    plt.axvline(512, color="red", linestyle="--", label="max_length = 512")
    plt.title(f"Textlängenverteilung – {strategy_name}")
    plt.xlabel("Wörter")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"text_lengths_{strategy_name}.png", dpi=150)
    plt.show()



# TF-IDF + Logistic Regression (Baseline-Modell)

def baseline_tfidf(train_texts, val_texts, train_labels, val_labels):

    print("\nBaseline: TF-IDF + Logistic Regression")

    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_val)

    metrics = {
        "accuracy":  accuracy_score(val_labels, preds),
        "f1":        f1_score(val_labels, preds, average="weighted"),
        "f1_macro":  f1_score(val_labels, preds, average="macro"),
        "precision": precision_score(val_labels, preds, average="weighted"),
        "recall":    recall_score(val_labels, preds, average="weighted"),
    }

    print(classification_report(val_labels, preds, target_names=["Fake", "Real"]))
    print(f" F1 (weighted): {metrics['f1']*100:.2f}%")
    print(f" F1 (macro)   : {metrics['f1_macro']*100:.2f}%")

    cm = confusion_matrix(val_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"])
    plt.title("Konfusionsmatrix – TF-IDF Baseline")
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.tight_layout()
    plt.savefig("confusion_baseline.png", dpi=150)
    plt.show()

    return metrics



# Metrics for HuggingFace train

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy":  accuracy_score(labels, predictions),
        "f1":        f1_score(labels, predictions, average="weighted"),
        "f1_macro":  f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall":    recall_score(labels, predictions, average="weighted"),
    }



# Training-Argumente (modellspezifisch)

def get_training_args(output_dir, is_llama=False):
    """
    Modellspezifische Learning Rates:
        BERT & RoBERTa : 5e-5  
        LLaMA + LoRA   : 2e-4  
        Weil LLaMA zu viel VRAM braucht:
        per_device_train_batch_size=4 if is_llama else 16,  
        per_device_eval_batch_size=4 if is_llama else 16,   
        gradient_accumulation_steps=4 if is_llama else 1, 
        
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4 if is_llama else 16,  
        per_device_eval_batch_size=4 if is_llama else 16,   
        gradient_accumulation_steps=4 if is_llama else 1,   
        learning_rate=2e-4 if is_llama else 5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,                         
    )

# LoRA Config (LLaMA)

def get_lora_config():
    """
    Skalierungsfaktor = 2
    """
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )



# Model train

def train_model(model_name, train_texts, val_texts,
                train_labels, val_labels, output_dir, max_length=512):
    print(f"\n Training: {model_name}")
    os.makedirs(output_dir, exist_ok=True)

    is_llama = "llama" in model_name.lower()

    #  AutoTokenizer für alle 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #  Model load 
    if "bert" in model_name.lower() and "roberta" not in model_name.lower():
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    elif "roberta" in model_name.lower():
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    else:
        # LLaMA Padding-Token bf16 LoRA
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(model, get_lora_config())
        model.print_trainable_parameters()

    model.to(device)

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset   = FakeNewsDataset(val_texts,   val_labels,   tokenizer, max_length)

    training_args = get_training_args(output_dir, is_llama=is_llama)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" Modell gespeichert: {output_dir}")

    return trainer, tokenizer



# Evaluation

def evaluate_model(trainer, tokenizer, val_texts, val_labels,
                   model_name, max_length=512):
    print(f"\n Evaluierung: {model_name}")

    val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, max_length)
    predictions_output = trainer.predict(val_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)

    metrics = {
        "accuracy":  accuracy_score(val_labels, preds),
        "f1":        f1_score(val_labels, preds, average="weighted"),
        "f1_macro":  f1_score(val_labels, preds, average="macro"),
        "precision": precision_score(val_labels, preds, average="weighted"),
        "recall":    recall_score(val_labels, preds, average="weighted"),
    }

    print(classification_report(val_labels, preds, target_names=["Fake", "Real"]))
    print(f"   F1 (weighted): {metrics['f1']*100:.2f}%")
    print(f"   F1 (macro)   : {metrics['f1_macro']*100:.2f}%")

    cm = confusion_matrix(val_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"])
    plt.title(f"Konfusionsmatrix – {model_name}")
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.tight_layout()
    plt.savefig(f"confusion_{model_name.replace('/', '_')}.png", dpi=150)
    plt.show()

    return metrics



# Visualization

def plot_model_comparison(results: dict, strategy: str = "title_and_content"):
    models    = list(results.keys())
    colors    = ["#95A5A6", "#4C72B0", "#DD8452", "#55A868"]
    x         = np.arange(len(models))
    bar_width = 0.3

    f1_vals       = [results[m]["f1"]      * 100 for m in models]
    f1_macro_vals = [results[m]["f1_macro"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_f1 = ax.bar(x - bar_width / 2, f1_vals, bar_width,
                     label="F1 (weighted)", color=colors, edgecolor="black", alpha=0.9)
    bars_fm = ax.bar(x + bar_width / 2, f1_macro_vals, bar_width,
                     label="F1 (macro)", color=colors, edgecolor="black",
                     alpha=0.5, hatch="//")

    for bar, val in zip(bars_f1, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_fm, f1_macro_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Modellvergleich inkl. Baseline – F1 Weighted vs. Macro\n"
                 f"Strategie: {strategy} | PUBHEALTH")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"model_comparison_{strategy}.png", dpi=150)
    plt.show()


def plot_full_metrics(results: dict, strategy: str = "title_and_content"):
    metric_keys   = ["accuracy", "f1", "f1_macro", "precision", "recall"]
    metric_labels = ["Accuracy", "F1\n(weighted)", "F1\n(macro)", "Precision", "Recall"]
    models  = list(results.keys())
    colors  = ["#95A5A6", "#4C72B0", "#DD8452", "#55A868"]
    x       = np.arange(len(metric_keys))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (model_name, color) in enumerate(zip(models, colors)):
        values = [results[model_name][k] * 100 for k in metric_keys]
        bars = ax.bar(x + i * width, values, width,
                      label=model_name, color=color, edgecolor="black")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Vollständiger Vergleich inkl. Baseline\n"
                 f"Strategie: {strategy} | PUBHEALTH")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"full_metrics_{strategy}.png", dpi=150)
    plt.show()



# MAIN

if __name__ == "__main__":

    MAX_LENGTH = 512

    MODELS = {
        "BERT"   : ("bert-base-uncased",       "output/bert"),
        "RoBERTa": ("roberta-base",             "output/roberta"),
        "LLaMA"  : ("meta-llama/Llama-3.2-3B", "output/llama"),
    }

    STRATEGIES = [
        "title_only",
        "content_only",
        "title_and_content",
    ]

    # load data
    df = load_dataset_tsv(folder=".")

    # Strategy train + eval
    all_results = {}

    for strategy in STRATEGIES:
        print(f"\n{'═'*60}")
        print(f" Strategie: {strategy}")
        print(f"{'═'*60}")

        texts, labels = preprocess(df, strategy)
        analyze_text_lengths(texts, strategy)

        # Gemeinsamer Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        strategy_results = {}

        # Baseline
        strategy_results["Baseline"] = baseline_tfidf(
            train_texts, val_texts, train_labels, val_labels
        )

        # Modelle
        for display_name, (model_name, base_output_dir) in MODELS.items():
            output_dir = f"{base_output_dir}/{strategy}"
            trainer, tokenizer = train_model(
                model_name,
                train_texts, val_texts,
                train_labels, val_labels,
                output_dir, MAX_LENGTH
            )
            metrics = evaluate_model(
                trainer, tokenizer,
                val_texts, val_labels,
                display_name, MAX_LENGTH
            )
            strategy_results[display_name] = metrics

        all_results[strategy] = strategy_results

        plot_model_comparison(strategy_results, strategy)
        plot_full_metrics(strategy_results, strategy)

    # Vergleich
    print(f"\n{'+'*75}")
    print(" Gesamtvergleich über alle Strategien:")
    print(f"{'+'*75}")

    for strategy, res in all_results.items():
        print(f"\n  [{strategy}]")
        print(f"  {'Modell':<12} {'Accuracy':>10} {'F1':>10} {'F1-Macro':>10} "
              f"{'Precision':>10} {'Recall':>10}")
        print(f"  {'─'*65}")
        for name, m in res.items():
            print(f"  {name:<12} "
                  f"{m['accuracy']*100:>9.2f}% "
                  f"{m['f1']*100:>9.2f}% "
                  f"{m['f1_macro']*100:>9.2f}% "
                  f"{m['precision']*100:>9.2f}% "
                  f"{m['recall']*100:>9.2f}%")

    print(f"\n{'+'*75}")
    
