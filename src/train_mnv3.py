import os, re, io, copy, glob, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms as T
import timm
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

# ---------- Utils ----------
def set_seed(seed=42):
    # Fija semillas para reproducibilidad en Python, NumPy y PyTorch (CPU/GPU)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def base_id(path):
    # Extrae un ID base del nombre del archivo para agrupar variantes del mismo caso (e.g., augmentaciones o multi-capturas)
    # Preferencia por patrón ISIC_####; si no existe, toma el primer token por '-' o '_'
    stem = Path(path).stem
    m = re.match(r'(ISIC_\d+)', stem)
    return m.group(1) if m else stem.split('-')[0].split('_')[0]

def collate_with_paths(batch):
    # Collate personalizado que conserva las rutas originales de las imágenes junto con (tensor, etiqueta)
    xs, ys, ps = zip(*batch)
    xs = torch.stack(xs); ys = torch.tensor(ys)
    return xs, ys, list(ps)

class PathImageFolder(datasets.ImageFolder):
    "Devuelve (img, label, path)"
    def __getitem__(self, index):
        # Sobrescribe para retornar también la ruta del archivo
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)
        return img, target, path

class TestFolder(torch.utils.data.Dataset):
    # Dataset para test que intenta respetar el orden y los nombres definidos en test.csv (si existe)
    def __init__(self, folder, tfm, id_list=None):
        self.folder = Path(folder)
        self.tfm = tfm
        exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}

        if id_list is not None:
            # Normaliza ids y resuelve nombres/extensión de forma robusta y en el MISMO orden del CSV
            id_norm = [str(x) for x in id_list if pd.notna(x)]
            self.paths = []
            for idname in id_norm:
                base = Path(idname)
                candidates = []
                if base.suffix:
                    # Si el id ya trae extensión, prueba ese nombre directamente
                    candidates = [base.name]
                else:
                    # Si no trae extensión, prueba variantes comunes
                    candidates = [f"{base.name}{suf}" for suf in ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG']]

                found = None
                # 1) match directo exacto
                for cn in candidates:
                    p = self.folder / cn
                    if p.exists():
                        found = p; break
                # 2) búsqueda case-insensitive si no se encontró
                if found is None:
                    low_targets = {cn.lower() for cn in candidates}
                    for p in self.folder.iterdir():
                        if p.suffix in exts and p.name.lower() in low_targets:
                            found = p; break
                if found is not None:
                    self.paths.append(str(found))
                else:
                    # Si no se encuentra, se avisa pero se continúa (deja hueco en submission si faltan)
                    print(f"[Warn] test id no encontrado en disco: {idname}")
        else:
            # Sin CSV: escanea la carpeta test/ (no recomendable si Kaggle exige orden específico)
            self.paths = sorted([str(p) for p in self.folder.iterdir() if p.suffix in exts])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        # Retorna imagen transformada y el nombre de archivo (para el CSV final)
        x = Image.open(self.paths[i]).convert('RGB')
        return self.tfm(x), os.path.basename(self.paths[i])

class RandomJPEGCompression:
    # Transformación de data augmentation que aplica compresión JPEG aleatoria (simula artefactos de compresión)
    def __init__(self, qmin=35, qmax=90, p=0.3):
        self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=random.randint(self.qmin, self.qmax)); buf.seek(0)
        return Image.open(buf).convert('RGB')

def make_dataloaders(data_dir, img_size, batch, workers):
    # Crea DataLoaders para train/valid/test con transformaciones y muestreo ponderado por clase y grupo
    train_dir = Path(data_dir)/'train'
    val_dir   = Path(data_dir)/'val'
    test_dir  = Path(data_dir)/'test'

    # Aumentaciones para entrenamiento + normalización ImageNet
    train_tfms = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.2), T.RandomRotation(15),
        RandomJPEGCompression(p=0.3),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    # Validación/Test: solo resize + normalización
    val_tfms = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Carga de datasets que además retornan el path de cada imagen
    train_ds = PathImageFolder(train_dir, transform=train_tfms)
    val_ds   = PathImageFolder(val_dir,   transform=val_tfms)

    # Reetiquetar mel->1 y nomel->0 (asegura consistencia sin depender del orden de carpetas)
    name_to_label = {'mel':1, 'nomel':0}
    def relabel(ds):
        new = []
        for p, y in ds.samples:
            cls = Path(p).parent.name.lower()
            new.append((p, name_to_label[cls]))
        ds.samples = new
        ds.targets = [y for _,y in ds.samples]
        ds.class_to_idx = {'nomel':0, 'mel':1}
    relabel(train_ds); relabel(val_ds)

    # Construcción de pesos por clase (balance) y por grupo (evitar sobreexposición de variantes del mismo caso)
    import collections
    train_groups = [base_id(p) for p,_ in train_ds.samples]
    class_counts = collections.Counter(train_ds.targets)
    group_counts = collections.Counter(train_groups)
    weights = []
    for (p,y), g in zip(train_ds.samples, train_groups):
        w_class = 1.0 / class_counts[y]   # balanceo de clases
        w_group = 1.0 / group_counts[g]   # balanceo por grupo/ID base
        weights.append(w_class * w_group)

    # Sampler ponderado para muestreo balanceado; DataLoaders con pin_memory y workers persistentes (si workers>0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    persistent = workers > 0

    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                          num_workers=workers, pin_memory=True, persistent_workers=persistent,
                          collate_fn=collate_with_paths)
    val_dl   = DataLoader(val_ds, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True, persistent_workers=persistent,
                          collate_fn=collate_with_paths)

    # Si existe data/test.csv, respeta ese orden de IDs; de lo contrario, se escanea test/
    test_csv = Path(data_dir)/'test.csv'
    id_list = None
    if test_csv.exists():
        try:
            dfids = pd.read_csv(test_csv)
            if 'id' in dfids.columns:
                id_list = [str(x) for x in dfids['id'] if pd.notna(x)]
            else:
                print("[Warn] test.csv no tiene columna 'id'; se usará escaneo de carpeta test/")
        except Exception as e:
            print(f"[Warn] No se pudo leer test.csv: {e}")

    # Dataset y DataLoader para test (sin etiquetas)
    test_ds  = TestFolder(test_dir, val_tfms, id_list=id_list)
    test_dl  = DataLoader(test_ds, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True, persistent_workers=persistent)

    # Diagnóstico: tamaños de splits y advertencia por archivos extra en test/ no listados en CSV
    n_train, n_val, n_test = len(train_ds.samples), len(val_ds.samples), len(test_ds)
    print(f"[Info] imgs -> train: {n_train} | valid: {n_val} | test usados: {n_test}")
    if id_list is not None:
        # Advierte si hay archivos extra en test/ que no están en el CSV (se ignoran para mantener orden)
        exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
        set_csv = set(x.lower() if Path(x).suffix else (x+'.jpg').lower() for x in id_list)
        extras = [p.name for p in test_dir.iterdir() if p.suffix in exts and p.name.lower() not in set_csv]
        if extras:
            print(f"[Warn] {len(extras)} archivos extra en test/ no están en test.csv (se IGNORAN). Ej: {extras[0]}")

    return train_dl, val_dl, test_dl, class_counts

@torch.no_grad()
def validate_groupwise(model, loader, device):
    # Valida agrupando por ID base (promedia probabilidades por grupo) para obtener una decisión por caso
    model.eval()
    probs, ys, gids = [], [], []
    for xb, yb, paths in loader:
        xb = xb.to(device).to(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            pr = torch.sigmoid(model(xb)).squeeze(1)
        probs.extend(pr.detach().cpu().numpy().tolist())
        ys.extend(yb.numpy().tolist())
        gids.extend([base_id(p) for p in paths])

    # Agrega probabilidades por grupo (ID base) y conserva la etiqueta del grupo
    by_group = {}
    for p, y, g in zip(probs, ys, gids):
        if g not in by_group: by_group[g] = {'probs':[], 'y': y}
        by_group[g]['probs'].append(p)
    g_probs = np.array([np.mean(v['probs']) for v in by_group.values()])
    g_y     = np.array([v['y'] for v in by_group.values()])

    # Búsqueda de umbral que maximiza F1 en una grilla [0.05, 0.95]
    thr_grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(g_y, (g_probs>=t).astype(int)) for t in thr_grid]
    i = int(np.argmax(f1s))
    best_thr, best_f1 = float(thr_grid[i]), float(f1s[i])

    # Métricas adicionales en el mejor umbral (precision, recall, matriz de confusión)
    yhat = (g_probs >= best_thr).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(g_y, yhat, average='binary', zero_division=0)
    cm = confusion_matrix(g_y, yhat)
    return best_f1, best_thr, (p, r, cm)

def build_model():
    # Crea MobileNetV3 grande preentrenada (ImageNet) para clasificación binaria (logit único)
    m = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=1)
    return m

def train_and_predict(args):
    # Orquestación: seeds, dataloaders, modelo, entrenamiento (cabeza + fine-tune), guardado y predicción en test
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carga DataLoaders y calcula pos_weight para BCE (balancea clases en la pérdida)
    train_dl, val_dl, test_dl, class_counts = make_dataloaders(args.data_dir, args.img_size, args.batch, args.workers)
    pos_weight = (class_counts[0] / class_counts[1]) if (1 in class_counts and class_counts[1]>0) else 1.0
    print(f"[Info] class_counts={dict(class_counts)}, pos_weight={pos_weight:.3f}")

    # Modelo (channels_last para optimizar memoria y throughput en GPU)
    model = build_model().to(device)
    model.to(memory_format=torch.channels_last)

    # AMP (GradScaler) si hay CUDA; optimizador AdamW; scheduler Cosine
    scaler = torch.amp.GradScaler('cuda') if device.type=='cuda' else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_epochs = args.epochs_head + args.epochs_ft
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,total_epochs))

    def run_epoch(loader, train=True, accumulation=1):
        # Bucle de entrenamiento/validación para una época con acumulación de gradientes
        if train: model.train()
        else: model.eval()
        total_loss, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for xb, yb, _ in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.float().unsqueeze(1).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(xb)
                loss = criterion(logits, yb) / accumulation
            if train:
                if scaler:
                    # Backprop con AMP
                    scaler.scale(loss).backward()
                    if (steps+1) % accumulation == 0:
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                else:
                    # Backprop en FP32
                    loss.backward()
                    if (steps+1) % accumulation == 0:
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * accumulation
            steps += 1
        return total_loss/max(1,steps)

    # Cabeza: congelar todo menos el clasificador y entrenar algunas épocas
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True

    best_f1, best_thr, best_state, wait = -1.0, 0.5, None, 0
    for e in range(args.epochs_head):
        tr_loss = run_epoch(train_dl, train=True, accumulation=args.accum)
        f1, thr, (p,r,cm) = validate_groupwise(model, val_dl, device)
        scheduler.step()
        if f1 > best_f1: best_f1, best_thr, best_state, wait = f1, thr, copy.deepcopy(model.state_dict()), 0
        else: wait += 1
        print(f"[HEAD] ep {e+1}/{args.epochs_head} | val_f1={f1:.4f} thr={thr:.3f} P={p:.3f} R={r:.3f} best={best_f1:.4f}")

    # Fine-tune: descongelar toda la red y continuar entrenando con early stopping por paciencia
    for p in model.parameters(): p.requires_grad = True
    for e in range(args.epochs_ft):
        tr_loss = run_epoch(train_dl, train=True, accumulation=args.accum)
        f1, thr, (p,r,cm) = validate_groupwise(model, val_dl, device)
        scheduler.step()
        if f1 > best_f1: best_f1, best_thr, best_state, wait = f1, thr, copy.deepcopy(model.state_dict()), 0
        else: wait += 1
        print(f"[FT]   ep {e+1}/{args.epochs_ft} | val_f1={f1:.4f} thr={thr:.3f} P={p:.3f} R={r:.3f} best={best_f1:.4f}")
        if wait >= args.patience: print("[Info] Early stopping."); break

    # Guardar mejor checkpoint (estado del modelo + mejor umbral y F1)
    Path("outputs").mkdir(exist_ok=True)
    ckpt = Path("outputs")/ "mnv3_best_f1.pth"
    torch.save({'state_dict': best_state, 'best_thr': best_thr, 'best_f1': best_f1}, ckpt)
    print(f"[Save] {ckpt} | best_f1={best_f1:.4f} best_thr={best_thr:.3f}")

    # Inferencia en test: aplica mejor umbral (best_thr) y genera CSV con columnas (id, prediction)
    model.load_state_dict(best_state); model.eval()
    ids, preds = [], []
    with torch.no_grad():
        for xb, names in test_dl:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                pr = torch.sigmoid(model(xb)).squeeze(1)
            binpred = (pr.detach().cpu().numpy() >= best_thr).astype(int).tolist()
            ids.extend(names); preds.extend(binpred)

    sub = pd.DataFrame({'id': ids, 'prediction': preds})
    out_csv = Path(args.outfile)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"[Save] {out_csv} (columnas: id,prediction | 0=No Mel, 1=Mel)")

def parse_args():
    # Parser de argumentos CLI con defaults pensados para la estructura local del proyecto
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='./data')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--epochs_head', type=int, default=2)
    ap.add_argument('--epochs_ft', type=int, default=15)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outfile', type=str, default='./data/sample_submission.csv')
    return ap.parse_args()

if __name__ == '__main__':
    # Punto de entrada: parsea argumentos y corre entrenamiento + predicción
    train_and_predict(parse_args())
