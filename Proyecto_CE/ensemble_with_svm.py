"""
Ensemble: RoBERTuito (fine-tuned) + BETO (fine-tuned) + SVM (embeddings RoBERTuito)

Estrategia de combinación: Soft-voting con pesos optimizados sobre validación.

Puntos clave:
- SVC necesita probability=True para emitir probabilidades calibradas (Platt scaling).
- Las probabilidades del SVM y de las redes están en el mismo espacio [0,1] → combinables.
- El SVM aporta diversidad estructural: optimiza margen en el espacio de embeddings,
  mientras que el fine-tuning optimiza cross-entropy end-to-end.

Requiere: pip install cma scikit-learn lightgbm transformers pysentimiento torch pandas scipy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
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

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN      = 64
BATCH_SIZE   = 16
EPOCHS       = 10
DATA_PATH    = "./Datasets/dataset_humor_train.json"
TEST_PATH    = "./Datasets/dataset_humor_test.json"
ROBERTUITO   = "pysentimiento/robertuito-base-cased"
BETO         = "dccuchile/bert-base-spanish-wwm-cased"

print(f"Dispositivo: {DEVICE}")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class HumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = preprocess_tweet(str(self.texts[idx]), lang="es")
        label = self.labels[idx]
        enc   = self.tokenizer(
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


# ─── Módulo de embeddings (Mean + Max pooling) ───────────────────────────────

class RoBERTuitoEmbeddings(nn.Module):
    """
    Extrae un vector de 768*2=1536 dimensiones via Mean+Max pooling.
    Usado tanto para el SVM como para el fine-tuning.
    """
    def __init__(self, model_id=ROBERTUITO):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)

    def forward(self, input_ids, attention_mask):
        out      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h        = out.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(h.size()).float()

        mean_pooled = torch.sum(h * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

        h_cloned = h.clone()
        h_cloned[mask_exp == 0] = -1e9
        max_pooled = torch.max(h_cloned, 1)[0]

        return torch.cat((mean_pooled, max_pooled), 1)   # [B, 1536]


# ─── Clasificador neuronal genérico (fine-tuning) ────────────────────────────

class HumorClassifier(nn.Module):
    def __init__(self, model_id, C1=512, C2=128, dropout1=0.4, dropout2=0.25):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_id)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, C1),
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
        out      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h        = out.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(h.size()).float()

        mean_pooled = torch.sum(h * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        h_cloned    = h.clone()
        h_cloned[mask_exp == 0] = -1e9
        max_pooled  = torch.max(h_cloned, 1)[0]

        return self.classifier(torch.cat((mean_pooled, max_pooled), 1))

    def get_probs(self, input_ids, attention_mask):
        return F.softmax(self.forward(input_ids, attention_mask), dim=1)


# ─── Entrenamiento fine-tuning ────────────────────────────────────────────────

def train_neural_model(model_id, save_path, X_tr, y_tr, X_val, y_val,
                       lr=2e-5, C1=512, C2=128, dropout1=0.4, dropout2=0.25,
                       weight_decay=1e-2, warmup_ratio=0.06) -> tuple:

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_id}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_dl  = DataLoader(HumorDataset(X_tr, y_tr, tokenizer),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl    = DataLoader(HumorDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = HumorClassifier(model_id, C1=C1, C2=C2, dropout1=dropout1, dropout2=dropout2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(total_steps * warmup_ratio),
                    num_training_steps=total_steps)
    loss_fn   = nn.CrossEntropyLoss()
    best_f1   = 0.0

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in train_dl:
            ids, mask, tgts = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            losses.append(loss.item())

        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids, mask, tgts = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
                _, p = torch.max(model(ids, mask), dim=1)
                preds.extend(p.cpu().tolist()); reals.extend(tgts.cpu().tolist())

        val_f1 = f1_score(reals, preds, average="macro")
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Loss: {np.mean(losses):.4f} | Val F1: {val_f1:.4f}", end="")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(" ★")
        else:
            print()

    print(f"  Mejor F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, tokenizer


# ─── Extracción de embeddings ─────────────────────────────────────────────────

def extract_embeddings(embed_model, dataloader, has_labels=True):
    """Extrae embeddings con el modelo de embeddings (no el clasificador)."""
    embed_model.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids  = batch["input_ids"].to(DEVICE) if has_labels else batch[0].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE) if has_labels else batch[1].to(DEVICE)
            emb  = embed_model(ids, mask)
            X_list.append(emb.cpu().numpy())
            if has_labels and "labels" in batch:
                y_list.extend(batch["labels"].numpy())
    X = np.vstack(X_list)
    y = np.array(y_list) if y_list else None
    return X, y


# ─── Entrenamiento SVM + clasificadores alternativos ─────────────────────────

def train_svm(X_tr, y_tr, X_val, y_val, C=1.5531, use_calibration=True):
    """
    Entrena SVM con kernel RBF.

    IMPORTANTE: probability=True activa Platt scaling (calibración interna via CV),
    que convierte los scores del SVM en probabilidades bien calibradas.
    Alternativa más rápida: CalibratedClassifierCV con cv=3.
    """
    print("\n" + "="*60)
    print("Entrenando SVM RBF (con calibración de probabilidades)")
    print("="*60)

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_val_s  = scaler.transform(X_val)

    if use_calibration:
        # CalibratedClassifierCV es más rápido que probability=True en SVC
        # porque calibra en un fold externo en lugar de re-entrenar 5 veces
        base_svm = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced")
        svm = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
    else:
        # probability=True usa Platt scaling interno (más lento, generalmente mejor)
        svm = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", probability=True)

    svm.fit(X_tr_s, y_tr)

    val_probs = svm.predict_proba(X_val_s)
    val_preds = np.argmax(val_probs, axis=1)
    val_f1    = f1_score(y_val, val_preds, average="macro")
    print(f"  SVM Val F1: {val_f1:.4f}")

    # Guardar
    joblib.dump({"svm": svm, "scaler": scaler}, "svm_model.pkl")
    print("  Guardado en: svm_model.pkl")
    return svm, scaler, val_f1


def train_lgbm(X_tr, y_tr, X_val, y_val):
    """
    LightGBM sobre embeddings — alternativa al SVM, más rápido y comparable en F1.
    Emite probabilidades directamente sin necesidad de calibración.
    """
    print("\n" + "="*60)
    print("Entrenando LightGBM sobre embeddings")
    print("="*60)

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    lgbm = lgb.LGBMClassifier(
        n_estimators    = 500,
        learning_rate   = 0.05,
        max_depth       = 6,
        num_leaves      = 63,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )
    lgbm.fit(
        X_tr_s, y_tr,
        eval_set=[(X_val_s, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    val_probs = lgbm.predict_proba(X_val_s)
    val_preds = np.argmax(val_probs, axis=1)
    val_f1    = f1_score(y_val, val_preds, average="macro")
    print(f"  LightGBM Val F1: {val_f1:.4f}")

    joblib.dump({"lgbm": lgbm, "scaler": scaler}, "lgbm_model.pkl")
    return lgbm, scaler, val_f1


# ─── Optimización de pesos del ensemble ──────────────────────────────────────

def optimize_weights(prob_list: list, y_true: np.ndarray,
                     names: list = None) -> np.ndarray:
    """
    Nelder-Mead con 20 reinicios para encontrar pesos óptimos de ensemble.
    Maximiza F1 macro sobre validación.
    """
    n = len(prob_list)

    def neg_f1(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-9)
        blended = sum(wi * pi for wi, pi in zip(w, prob_list))
        return -f1_score(y_true, np.argmax(blended, axis=1), average="macro")

    best = None
    np.random.seed(42)
    for _ in range(30):
        x0  = np.random.dirichlet(np.ones(n))
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res

    w = np.clip(best.x, 0, None)
    w = w / w.sum()

    if names:
        print("\nPesos óptimos del ensemble:")
        for name, wi in zip(names, w):
            print(f"  {name:30s}: {wi:.4f}")

    return w


# ─── Obtener probabilidades de validación de modelos neurales ─────────────────

def get_neural_probs(model, tokenizer, X_val, y_val) -> np.ndarray:
    val_dl = DataLoader(HumorDataset(X_val, y_val, tokenizer),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in val_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            p    = model.get_probs(ids, mask)
            probs.append(p.cpu().numpy())
    return np.vstack(probs)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    # 1. Datos
    print("Cargando datos...")
    df        = pd.read_json(DATA_PATH, lines=True)
    textos    = df["text"].values
    etiquetas = df["klass"].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        textos, etiquetas, test_size=0.15, stratify=etiquetas, random_state=42
    )
    print(f"Train: {len(X_tr)} | Val: {len(X_val)}")

    # ── BLOQUE 1: Fine-tuning neurales ───────────────────────────────────────

    # RoBERTuito
    # Si tienes cmaes_best_params.json, carga los hiperparámetros aquí:
    robertuito_kwargs = dict(lr=2e-5, C1=512, C2=128, dropout1=0.40, dropout2=0.25,
                             weight_decay=1e-2, warmup_ratio=0.06)
    if os.path.exists("cmaes_best_params.json"):
        with open("cmaes_best_params.json") as f:
            cmaes = json.load(f)["params"]
        robertuito_kwargs.update({k: cmaes[k] for k in robertuito_kwargs if k in cmaes})
        print("Parámetros CMA-ES cargados para RoBERTuito")

    model_rbt, tok_rbt = train_neural_model(
        ROBERTUITO, "best_robertuito.pth", X_tr, y_tr, X_val, y_val, **robertuito_kwargs
    )
    torch.cuda.empty_cache()

    # BETO
    model_beto, tok_beto = train_neural_model(
        BETO, "best_beto.pth", X_tr, y_tr, X_val, y_val,
        lr=3e-5, C1=512, C2=128, dropout1=0.35, dropout2=0.20,
        weight_decay=1e-2, warmup_ratio=0.06
    )
    torch.cuda.empty_cache()

    # ── BLOQUE 2: SVM con embeddings de RoBERTuito ───────────────────────────

    print("\n" + "="*60)
    print("Extrayendo embeddings para SVM / LightGBM")
    print("="*60)

    # Usamos el encoder de RoBERTuito YA ENTRENADO (fine-tuned)
    # Esto captura representaciones más ricas que el modelo base
    embed_model = RoBERTuitoEmbeddings(ROBERTUITO).to(DEVICE)
    # Copiar pesos del encoder fine-tuneado directamente.
    # model_rbt.encoder es el AutoModel completo; embed_model.encoder es el mismo objeto.
    # load_state_dict sobre el módulo completo evita el mismatch de prefijos.
    embed_model.encoder.load_state_dict(model_rbt.encoder.state_dict())
    embed_model.eval()

    tokenizer_rbt = AutoTokenizer.from_pretrained(ROBERTUITO)
    train_dl_emb  = DataLoader(HumorDataset(X_tr,  y_tr,  tokenizer_rbt),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_dl_emb    = DataLoader(HumorDataset(X_val, y_val, tokenizer_rbt),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Extrayendo embeddings TRAIN...")
    X_tr_emb,  y_tr_emb  = extract_embeddings(embed_model, train_dl_emb)
    print(f"  Shape: {X_tr_emb.shape}")
    print("Extrayendo embeddings VAL...")
    X_val_emb, y_val_emb = extract_embeddings(embed_model, val_dl_emb)
    print(f"  Shape: {X_val_emb.shape}")

    # SVM
    svm_model, svm_scaler, svm_f1 = train_svm(X_tr_emb, y_tr_emb, X_val_emb, y_val_emb, C=1.5531)

    # LightGBM (opcional pero recomendado como 4to modelo si tienes recursos)
    lgbm_model, lgbm_scaler, lgbm_f1 = train_lgbm(X_tr_emb, y_tr_emb, X_val_emb, y_val_emb)

    # ── BLOQUE 3: Probabilidades en validación ────────────────────────────────

    print("\n" + "="*60)
    print("Recopilando probabilidades de validación")
    print("="*60)

    probs_rbt  = get_neural_probs(model_rbt, tok_rbt, X_val, y_val)
    probs_beto = get_neural_probs(model_beto, tok_beto, X_val, y_val)

    X_val_s_svm  = svm_scaler.transform(X_val_emb)
    probs_svm    = svm_model.predict_proba(X_val_s_svm)

    X_val_s_lgbm = lgbm_scaler.transform(X_val_emb)
    probs_lgbm   = lgbm_model.predict_proba(X_val_s_lgbm)

    # F1 individuales
    for name, probs in [("RoBERTuito FT", probs_rbt), ("BETO FT", probs_beto),
                        ("SVM", probs_svm), ("LightGBM", probs_lgbm)]:
        f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro")
        print(f"  {name:20s}: F1 = {f1:.4f}")

    # ── BLOQUE 4: Combinaciones del ensemble ──────────────────────────────────

    print("\n" + "="*60)
    print("Optimizando combinaciones de ensemble")
    print("="*60)

    # Definir qué combinaciones evaluar
    combos = {
        "RoBERTuito + SVM":             [probs_rbt, probs_svm],
        "RoBERTuito + BETO":            [probs_rbt, probs_beto],
        "RoBERTuito + BETO + SVM":      [probs_rbt, probs_beto, probs_svm],
        "RoBERTuito + BETO + LightGBM": [probs_rbt, probs_beto, probs_lgbm],
        "Todos (4 modelos)":            [probs_rbt, probs_beto, probs_svm, probs_lgbm],
    }

    results = {}
    for combo_name, prob_list in combos.items():
        # Pesos iguales
        blended_eq = sum(p / len(prob_list) for p in prob_list)
        f1_eq = f1_score(y_val, np.argmax(blended_eq, axis=1), average="macro")

        # Pesos optimizados
        w_opt     = optimize_weights(prob_list, y_val)
        blended_w = sum(wi * pi for wi, pi in zip(w_opt, prob_list))
        f1_w      = f1_score(y_val, np.argmax(blended_w, axis=1), average="macro")

        results[combo_name] = {"f1_equal": f1_eq, "f1_optimized": f1_w, "weights": w_opt.tolist()}
        print(f"  {combo_name}")
        print(f"    Pesos iguales: {f1_eq:.4f}  |  Pesos óptimos: {f1_w:.4f}")

    # ── BLOQUE 5: Mejor combinación ───────────────────────────────────────────

    best_combo = max(results, key=lambda k: results[k]["f1_optimized"])
    best_f1    = results[best_combo]["f1_optimized"]

    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Baseline (RoBERTuito solo): 0.8744")
    print(f"  SVM standalone:             {svm_f1:.4f}")
    print(f"  Mejor combinación:          [{best_combo}]  F1 = {best_f1:.4f}")
    delta = best_f1 - 0.8744
    print(f"  Mejora sobre baseline:      {delta:+.4f}")
    print("="*60)

    # Guardar resultados
    with open("ensemble_results.json", "w") as f:
        json.dump({"combos": results, "best_combo": best_combo, "best_f1": best_f1}, f, indent=2)
    print("Resultados guardados en: ensemble_results.json")

    # ── BLOQUE 6: Inferencia en test con la mejor combinación ─────────────────

    print(f"\n{'='*60}")
    print(f"INFERENCIA EN TEST — usando: {best_combo}")
    print("="*60)

    best_weights   = np.array(results[best_combo]["weights"])
    combo_models   = combos[best_combo]  # prob arrays (validación), no útiles para test
    combo_names    = best_combo.split(" + ")

    # Extraer probabilidades en test para cada modelo del mejor combo
    df_test   = pd.read_json(TEST_PATH, lines=True)
    raw_texts = df_test["text"].tolist()
    texts_pp  = [preprocess_tweet(t, lang="es") for t in raw_texts]

    all_test_probs = []

    # Modelos neurales
    for m_name, (model, tok) in [("RoBERTuito FT", (model_rbt, tok_rbt)),
                                   ("BETO FT",       (model_beto, tok_beto))]:
        if not any(n in best_combo for n in ["RoBERTuito", "BETO"]):
            continue
        if m_name.replace(" FT", "") not in best_combo and m_name not in best_combo:
            continue

        tokens = tok(texts_pp, add_special_tokens=True, max_length=MAX_LEN,
                     padding="max_length", truncation=True, return_tensors="pt")
        ds     = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
        dl     = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        t_probs = []
        with torch.no_grad():
            for batch in dl:
                ids, mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
                t_probs.append(model.get_probs(ids, mask).cpu().numpy())
        all_test_probs.append(np.vstack(t_probs))
        print(f"  {m_name}: ✓")

    # Embeddings para SVM / LightGBM
    tokens_emb = tokenizer_rbt(texts_pp, add_special_tokens=True, max_length=MAX_LEN,
                                padding="max_length", truncation=True, return_tensors="pt")
    ds_emb     = TensorDataset(tokens_emb["input_ids"], tokens_emb["attention_mask"])
    dl_emb     = DataLoader(ds_emb, batch_size=BATCH_SIZE, shuffle=False)

    embed_model.eval()
    test_embs = []
    with torch.no_grad():
        for batch in dl_emb:
            ids, mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
            test_embs.append(embed_model(ids, mask).cpu().numpy())
    X_test_emb = np.vstack(test_embs)

    if "SVM" in best_combo:
        X_test_s = svm_scaler.transform(X_test_emb)
        all_test_probs.append(svm_model.predict_proba(X_test_s))
        print("  SVM: ✓")

    if "LightGBM" in best_combo:
        X_test_s = lgbm_scaler.transform(X_test_emb)
        all_test_probs.append(lgbm_model.predict_proba(X_test_s))
        print("  LightGBM: ✓")

    # Ensemble final
    blended_test = sum(w * p for w, p in zip(best_weights, all_test_probs))
    predictions  = np.argmax(blended_test, axis=1)

    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nDistribución de clases: {dict(zip(unique.tolist(), counts.tolist()))}")

    pd.DataFrame({"prediction": predictions}).to_csv("predictions_ensemble_svm.csv", index=False)
    print("Predicciones guardadas en: predictions_ensemble_svm.csv")

    return predictions


if __name__ == "__main__":
    predictions = main()
