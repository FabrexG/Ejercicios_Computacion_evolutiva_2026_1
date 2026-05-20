"""
Ensemble RoBERTuito + BETO para detección de humor en español
Estrategia: Soft-voting con pesos aprendidos (weighted average de probabilidades)

Modelos:
- RoBERTuito (pysentimiento/robertuito-base-cased): especializado en tweets en español
- BETO (dccuchile/bert-base-spanish-wwm-cased):    BERT entrenado en español general
- Opcionalmente: BETO-uncased (dccuchile/bert-base-spanish-wwm-uncased)

La diversidad entre modelos (tweet-optimized vs general) reduce la varianza del ensemble.

Requiere: pip install transformers pysentimiento scikit-learn torch pandas scipy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import minimize
from pysentimiento.preprocessing import preprocess_tweet


# ─── Reproducibilidad ────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

# ─── Configuración ───────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 100
BATCH_SIZE = 16
EPOCHS     = 10
DATA_PATH  = "./Datasets/dataset_humor_train.json"
TEST_PATH  = "./Datasets/dataset_humor_test.json"

# Modelos a ensamblar
MODEL_CONFIGS = [
    {
        "name":       "robertuito",
        "model_id":   "pysentimiento/robertuito-base-cased",
        "save_path":  "best_robertuito.pth",
        # Parámetros — idealmente los mejores de CMA-ES; aquí defaults razonables
        "lr":         2e-5,
        "dropout1":   0.40,
        "dropout2":   0.25,
        "C1":         512,
        "C2":         128,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
    },
    {
        "name":       "beto",
        "model_id":   "dccuchile/bert-base-spanish-wwm-cased",
        "save_path":  "best_beto.pth",
        # BETO suele funcionar mejor con LR ligeramente mayor que RoBERTuito
        "lr":         3e-5,
        "dropout1":   0.35,
        "dropout2":   0.20,
        "C1":         512,
        "C2":         128,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
    },
    # Descomenta para agregar un tercer modelo (aumenta diversidad pero requiere más VRAM)
    # {
    #     "name":       "beto-uncased",
    #     "model_id":   "dccuchile/bert-base-spanish-wwm-uncased",
    #     "save_path":  "best_beto_uncased.pth",
    #     "lr":         3e-5,
    #     "dropout1":   0.35,
    #     "dropout2":   0.20,
    #     "C1":         512,
    #     "C2":         128,
    #     "weight_decay": 1e-2,
    #     "warmup_ratio": 0.06,
    # },
]

print(f"Dispositivo: {DEVICE}")
print(f"Modelos en el ensemble: {[m['name'] for m in MODEL_CONFIGS]}")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class HumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN, preprocess=True):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.preprocess = preprocess

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.preprocess:
            text = preprocess_tweet(text, lang="es")
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ─── Modelo base ─────────────────────────────────────────────────────────────

class HumorClassifier(nn.Module):
    """
    Clasificador genérico compatible con RoBERTuito, BETO y otros modelos BERT-like.
    Usa Mean+Max pooling sobre la capa oculta final.
    """
    def __init__(self, model_id, C1=512, C2=128, dropout1=0.4, dropout2=0.25):
        super().__init__()
        print(f"  Cargando {model_id}...")
        self.encoder = AutoModel.from_pretrained(model_id)
        input_dim = 768 * 2  # mean + max pooling

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, C1),
            nn.BatchNorm1d(C1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(C1, C2),
            nn.BatchNorm1d(C2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(C2, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h   = out.last_hidden_state

        mask_exp     = attention_mask.unsqueeze(-1).expand(h.size()).float()
        mean_pooled  = torch.sum(h * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

        h_cloned = h.clone()
        h_cloned[mask_exp == 0] = -1e9
        max_pooled = torch.max(h_cloned, 1)[0]

        concat = torch.cat((mean_pooled, max_pooled), 1)
        return self.classifier(concat)

    def get_probs(self, input_ids, attention_mask):
        """Retorna probabilidades softmax (útil para ensemble)."""
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=1)


# ─── Entrenamiento individual ─────────────────────────────────────────────────

def train_single_model(config, X_tr, y_tr, X_val, y_val) -> nn.Module:
    """Entrena un modelo con la configuración dada y retorna el mejor checkpoint."""
    name      = config["name"]
    model_id  = config["model_id"]
    save_path = config["save_path"]

    print(f"\n{'='*60}")
    print(f"Entrenando: {name.upper()}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = HumorDataset(X_tr, y_tr, tokenizer)
    val_ds   = HumorDataset(X_val, y_val, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = HumorClassifier(
        model_id  = model_id,
        C1        = config["C1"],
        C2        = config["C2"],
        dropout1  = config["dropout1"],
        dropout2  = config["dropout2"],
    ).to(DEVICE)

    optimizer     = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps   = len(train_dl) * EPOCHS
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        # Entrenamiento
        model.train()
        losses = []
        for batch in train_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tgts = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = loss_fn(logits, tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        # Validación
        model.eval()
        preds, reals, val_losses = [], [], []
        with torch.no_grad():
            for batch in val_dl:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tgts = batch["labels"].to(DEVICE)
                logits = model(ids, mask)
                val_losses.append(loss_fn(logits, tgts).item())
                _, p = torch.max(logits, dim=1)
                preds.extend(p.cpu().tolist())
                reals.extend(tgts.cpu().tolist())

        val_f1 = f1_score(reals, preds, average="macro")
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {np.mean(losses):.4f} | "
              f"Val Loss: {np.mean(val_losses):.4f} | "
              f"Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(" ★")
        else:
            print()

    print(f"\n  {name} — Mejor F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, tokenizer


# ─── Extraer probabilidades de validación ────────────────────────────────────

def get_val_probs(model, tokenizer, X_val, y_val) -> np.ndarray:
    """Retorna matriz [n_samples, 2] de probabilidades en el conjunto de validación."""
    val_ds = HumorDataset(X_val, y_val, tokenizer)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in val_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            probs = model.get_probs(ids, mask)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


# ─── Optimización de pesos del ensemble ──────────────────────────────────────

def optimize_ensemble_weights(prob_list: list, y_true: np.ndarray) -> np.ndarray:
    """
    Encuentra los pesos óptimos para el ensemble mediante optimización Nelder-Mead.
    Maximiza F1 macro sobre la validación.

    Args:
        prob_list: lista de arrays [n_samples, 2], uno por modelo
        y_true:    etiquetas reales
    Returns:
        pesos normalizados (suma = 1)
    """
    n_models = len(prob_list)

    def neg_f1(weights):
        weights = np.array(weights)
        weights = np.clip(weights, 0, None)
        weights = weights / (weights.sum() + 1e-9)
        blended = sum(w * p for w, p in zip(weights, prob_list))
        preds   = np.argmax(blended, axis=1)
        return -f1_score(y_true, preds, average="macro")

    # Punto de inicio: pesos iguales
    x0 = np.ones(n_models) / n_models

    # Múltiples reinicios para evitar mínimos locales
    best_res = None
    np.random.seed(42)
    for _ in range(20):
        x_init = np.random.dirichlet(np.ones(n_models))
        res = minimize(neg_f1, x_init, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    optimal_weights = np.array(best_res.x)
    optimal_weights = np.clip(optimal_weights, 0, None)
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    # 1. Cargar datos
    print("Cargando datos...")
    df        = pd.read_json(DATA_PATH, lines=True)
    textos    = df["text"].values
    etiquetas = df["klass"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        textos, etiquetas, test_size=0.15, stratify=etiquetas, random_state=42
    )
    print(f"Train: {len(X_tr)} | Val: {len(X_val)}")

    # 2. Cargar mejores parámetros de CMA-ES si existen
    if os.path.exists("cmaes_best_params.json"):
        print("\nCargando parámetros optimizados por CMA-ES...")
        with open("cmaes_best_params.json") as f:
            cmaes = json.load(f)
        cmaes_params = cmaes["params"]
        # Aplicar al primer modelo (RoBERTuito)
        for k in ["lr", "dropout1", "dropout2", "weight_decay", "warmup_ratio", "C1", "C2"]:
            MODEL_CONFIGS[0][k] = cmaes_params[k]
        print(f"Parámetros CMA-ES aplicados a {MODEL_CONFIGS[0]['name']}")
        print(f"  F1 reportado por CMA-ES: {cmaes.get('f1_full_training', cmaes['f1']):.4f}")

    # 3. Entrenar cada modelo individualmente
    trained_models    = []
    trained_tokenizers = []

    for config in MODEL_CONFIGS:
        model, tok = train_single_model(config, X_tr, y_tr, X_val, y_val)
        trained_models.append(model)
        trained_tokenizers.append(tok)
        torch.cuda.empty_cache()

    # 4. Obtener probabilidades en validación para cada modelo
    print("\nObteniendo probabilidades de validación para calibración del ensemble...")
    prob_list = []
    individual_f1s = []

    for i, (model, tok) in enumerate(zip(trained_models, trained_tokenizers)):
        name  = MODEL_CONFIGS[i]["name"]
        probs = get_val_probs(model, tok, X_val, y_val)
        preds = np.argmax(probs, axis=1)
        f1    = f1_score(y_val, preds, average="macro")
        print(f"  {name}: F1 = {f1:.4f}")
        prob_list.append(probs)
        individual_f1s.append(f1)

    # 5. Ensemble simple (pesos iguales) — baseline
    blended_equal = sum(p / len(prob_list) for p in prob_list)
    preds_equal   = np.argmax(blended_equal, axis=1)
    f1_equal      = f1_score(y_val, preds_equal, average="macro")
    print(f"\nEnsemble pesos iguales: F1 = {f1_equal:.4f}")

    # 6. Optimizar pesos del ensemble
    print("\nOptimizando pesos del ensemble...")
    optimal_weights = optimize_ensemble_weights(prob_list, y_val)
    print(f"Pesos óptimos: " + ", ".join(
        f"{MODEL_CONFIGS[i]['name']}={w:.3f}" for i, w in enumerate(optimal_weights)
    ))

    blended_opt = sum(w * p for w, p in zip(optimal_weights, prob_list))
    preds_opt   = np.argmax(blended_opt, axis=1)
    f1_opt      = f1_score(y_val, preds_opt, average="macro")
    print(f"Ensemble pesos óptimos: F1 = {f1_opt:.4f}")

    # 7. Resumen
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    for i, f1 in enumerate(individual_f1s):
        print(f"  {MODEL_CONFIGS[i]['name']:20s}: {f1:.4f}")
    print(f"  {'Ensemble (igual)':20s}: {f1_equal:.4f}")
    print(f"  {'Ensemble (óptimo)':20s}: {f1_opt:.4f}")
    print(f"  {'Baseline (0.8744)':20s}: 0.8744")
    delta = f1_opt - 0.8744
    print(f"  Mejora sobre baseline: {delta:+.4f}")
    print("="*60)

    # 8. Guardar pesos del ensemble
    ensemble_info = {
        "model_names":       [c["name"] for c in MODEL_CONFIGS],
        "individual_f1s":    individual_f1s,
        "ensemble_f1_equal": f1_equal,
        "ensemble_f1_opt":   f1_opt,
        "optimal_weights":   optimal_weights.tolist(),
    }
    with open("ensemble_weights.json", "w") as f:
        json.dump(ensemble_info, f, indent=2)
    print("Pesos guardados en: ensemble_weights.json")

    # 9. Inferencia en test
    print(f"\n{'='*60}")
    print("INFERENCIA EN TEST")
    print("="*60)

    df_test   = pd.read_json(TEST_PATH, lines=True)
    raw_texts = df_test["text"].tolist()

    all_test_probs = []
    for i, (model, tok) in enumerate(zip(trained_models, trained_tokenizers)):
        name = MODEL_CONFIGS[i]["name"]
        print(f"  Inferencia {name}...")
        texts_processed = [preprocess_tweet(t, lang="es") for t in raw_texts]

        tokens = tok(
            texts_processed,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        ds  = torch.utils.data.TensorDataset(tokens["input_ids"], tokens["attention_mask"])
        dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        test_probs = []
        with torch.no_grad():
            for batch in dl:
                ids  = batch[0].to(DEVICE)
                mask = batch[1].to(DEVICE)
                probs = model.get_probs(ids, mask)
                test_probs.append(probs.cpu().numpy())

        all_test_probs.append(np.vstack(test_probs))

    # Ensemble con pesos óptimos
    blended_test = sum(w * p for w, p in zip(optimal_weights, all_test_probs))
    predictions  = np.argmax(blended_test, axis=1)

    print("\nPredicciones generadas.")
    unique, counts = np.unique(predictions, return_counts=True)
    print("Distribución de clases:", dict(zip(unique.tolist(), counts.tolist())))

    # Guardar predicciones
    df_out = pd.DataFrame({"prediction": predictions})
    df_out.to_csv("predictions_ensemble.csv", index=False)
    print("Predicciones guardadas en: predictions_ensemble.csv")

    return predictions, ensemble_info


if __name__ == "__main__":
    predictions, info = main()
