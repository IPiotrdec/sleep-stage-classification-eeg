import os
import numpy as np
import mne
from scipy.signal import welch  # metoda Welcha do PSD 
import random

random.seed(42)     # seed dla powtarzalności
np.random.seed(42)

DATA_DIR = r"data" # katalog z plikami .edf

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # stara N4 liczy się dziś jako N3
    "Sleep stage R": 4,
}

PICKS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"] # lista kanałów do użycia, dodany EOG do poprawy klasyfikacji REM

#trapz na wersji 3.14.0
def extract_epoch_features(epoch, sfreq):
    f, Pxx = welch(epoch, fs=sfreq, nperseg=min(len(epoch), int(4 * sfreq)))

    def bandpower(fmin, fmax):
        idx = (f >= fmin) & (f < fmax)
        if not np.any(idx):
            return 0.0
        return float(np.trapz(Pxx[idx], f[idx])) 

    delta = bandpower(0.5, 4.0)
    theta = bandpower(4.0, 8.0)
    alpha = bandpower(8.0, 12.0)
    beta  = bandpower(12.0, 30.0)

    total_power = delta + theta + alpha + beta + 1e-12

    delta_rel = delta / total_power
    theta_rel = theta / total_power
    alpha_rel = alpha / total_power
    beta_rel  = beta  / total_power


    std_val  = float(np.std(epoch))
    max_abs  = float(np.max(np.abs(epoch)))

    return np.array([delta_rel, theta_rel, alpha_rel, beta_rel, std_val, max_abs], dtype=float)

# Przetwarzanie jednego rekordu (PSG + hypnogram)
def process_one_record(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")

    # Filtr 0.5–30 Hz...      w polsce używa się 50 Hz, ale zbiór jest zagraniczny
    raw.filter(l_freq=0.5, h_freq=30.0, verbose="ERROR")

    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann)

    events, event_id = mne.events_from_annotations(raw, event_id=STAGE_MAP)

    sfreq = raw.info["sfreq"]
    epoch_length = 30.0
    tmin = 0.0
    tmax = epoch_length - 1.0 / sfreq # sfreq = 100, bez -1/sfreq byłoby powyżej 30s, a ma się zmieścić w 30s

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR"
    )

    ch_present = set(epochs.ch_names)
    if not all(ch in ch_present for ch in PICKS):
        print(f"  UWAGA: pomijam {os.path.basename(psg_path)} - brak kanałów z listy PICKS.")
        return None, None

    epochs_data = epochs.copy().pick(PICKS).get_data()  # (n_epochs, n_channels, n_samples)
    n_epochs, n_channels, _ = epochs_data.shape
    
    y = epochs.events[:, 2]

    features = []
    for ep_idx in range(n_epochs):
        feats_ep = []
        for ch_idx in range(n_channels):
            ep_signal = epochs_data[ep_idx, ch_idx, :]
            feats_ch = extract_epoch_features(ep_signal, sfreq)
            feats_ep.append(feats_ch)
        features.append(np.concatenate(feats_ep))

    X_features = np.vstack(features)
    return X_features, y

# Zbieranie danych ze wszystkich rekordów w katalogu
def collect_all_records(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".edf"))

    records = {}
    for fname in files:
        base = fname[:6]  # np. SC4001
        records.setdefault(base, []).append(fname)

    X_all_list, y_all_list, g_all_list = [], [], []

    for base, fnames in records.items():
        psg = [f for f in fnames if "PSG" in f]
        hyp = [f for f in fnames if "Hypnogram" in f]

        if len(psg) != 1 or len(hyp) != 1:
            print(f"Pominięto {base}: PSG={psg}, Hyp={hyp}")
            continue

        psg_path = os.path.join(data_dir, psg[0])
        hyp_path = os.path.join(data_dir, hyp[0])

        print(f"Przetwarzam rekord: {base}")
        X_rec, y_rec = process_one_record(psg_path, hyp_path)

        if X_rec is None:
            continue

        print(f"  epok: {X_rec.shape[0]}, cech/epokę: {X_rec.shape[1]}")

        X_all_list.append(X_rec)
        y_all_list.append(y_rec)
        g_all_list.append(np.full(len(y_rec), base))  # group id dla każdej epoki

    if not X_all_list:
        raise RuntimeError("Nie znaleziono żadnych poprawnych par PSG/Hypnogram.")

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)
    groups_all = np.concatenate(g_all_list)

    return X_all, y_all, groups_all

if __name__ == "__main__":
    X_all, y_all, groups_all = collect_all_records(DATA_DIR)

    print("\n=== PODSUMOWANIE ===")
    print("Łączna liczba epok:", X_all.shape[0])
    print("Liczba cech na epokę:", X_all.shape[1])

    unique, counts = np.unique(y_all, return_counts=True)
    print("Unikalne etykiety:", dict(zip(unique, counts)))

    # Model / metryki
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier

    # Split po REKORDACH (bez leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Pipeline standaryzacja + k-NN
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=11, weights="distance")
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("\n=== WYNIKI K-NN (Group split) ===")
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))




########################################################
#Wykresy pomocnicze do analizy + zapisywanie do plików##
########################################################
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Nazwy klas
class_names = ["W (0)", "N1 (1)", "N2 (2)", "N3 (3)", "REM (4)"]

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])

# macierz pomyłek
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Counts)")
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")

# liczby w komórkach
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_counts.png", dpi=200)

# Normalizacja
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

plt.figure()
plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
plt.title("Confusion Matrix (Normalized)")
plt.colorbar()
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_norm.png", dpi=200)

# wyniki na klasę
prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=[0,1,2,3,4], zero_division=0
)

x = np.arange(len(class_names))
w = 0.25

plt.figure()
plt.bar(x - w, prec, width=w, label="Precision")
plt.bar(x,      rec,  width=w, label="Recall")
plt.bar(x + w,  f1,   width=w, label="F1-Score")
plt.xticks(x, class_names, rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("Quality of Classification by Class")
plt.legend()
plt.tight_layout()
plt.savefig("metrics.png", dpi=200)

# liczba epok na klasę
plt.figure()
plt.bar(class_names, sup)
plt.title("Number of epochs per class (test set)")
plt.ylabel("Number of epochs")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("support_test.png", dpi=200)

plt.show()

