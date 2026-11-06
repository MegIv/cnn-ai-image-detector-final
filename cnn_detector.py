"""CNN CIFAKE: REAL=0, FAKE=1. Jalankan --help untuk penggunaan CLI."""

# %% IMPORTS
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

LABELS = {"REAL": 0, "FAKE": 1}
DATASET_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"


@dataclass
class Config:
    seed: int = 42
    image_size: int = 32
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 0.001
    validation_fraction: float = 0.2
    threshold: float = 0.5

    def validate(self):
        if self.image_size != 32:
            raise ValueError("Eksperimen ini menggunakan CIFAKE asli berukuran 32x32.")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("Batch size, epochs, dan learning rate harus positif.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction harus di antara 0 dan 1.")
        if self.threshold != 0.5:
            raise ValueError("Baseline memakai threshold tetap 0.5, ditetapkan sebelum tes.")


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# %% DATA
def find_dataset_root(path):
    """Mendukung folder CIFAKE langsung maupun satu/beberapa folder pembungkus."""
    path = Path(path).expanduser().resolve()
    candidates = [path] + sorted(p.parent for p in path.rglob("train") if p.is_dir())
    roots = []
    for candidate in candidates:
        if all((candidate / split / label).is_dir()
               for split in ("train", "test") for label in LABELS):
            if candidate not in roots:
                roots.append(candidate)
    if len(roots) != 1:
        raise ValueError(f"Harus ada tepat satu folder train/REAL, train/FAKE, "
                         f"test/REAL, test/FAKE. Ditemukan {len(roots)} di {path}.")
    return roots[0]


def download_dataset(cache_dir="data/cache"):
    """Unduh dari sumber penulis. Cache tetap di folder proyek."""
    os.environ["KAGGLEHUB_CACHE"] = str(Path(cache_dir).resolve())
    import kagglehub
    return find_dataset_root(kagglehub.dataset_download(DATASET_HANDLE))


def inspect_image(item):
    path, root, split, label = item
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            width, height = rgb.size
            digest = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        return {"path": path.relative_to(root).as_posix(), "source_split": split,
                "class_name": label, "label": LABELS[label], "sha256_pixels": digest,
                "width": width, "height": height, "error": ""}
    except Exception as exc:
        return {"path": path.relative_to(root).as_posix(), "error": str(exc)}


def prepare_splits(root, config, output_dir):
    """Audit piksel identik, sisihkan overlap dari train, lalu split terstratifikasi."""
    root = find_dataset_root(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for split in ("train", "test"):
        for label in LABELS:
            files = sorted(p for p in (root / split / label).rglob("*")
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
            if not files:
                raise ValueError(f"Folder kosong: {split}/{label}")
            items.extend((p, root, split, label) for p in files)
    print(f"Audit {len(items):,} gambar; proses ini dapat memerlukan beberapa menit.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(inspect_image, items))
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    invalid = manifest[manifest.error != ""]
    if len(invalid):
        raise ValueError(f"{len(invalid)} gambar gagal dibaca; periksa source_manifest.csv.")
    if not ((manifest.width == 32) & (manifest.height == 32)).all():
        raise ValueError("Dataset harus berisi citra 32x32; lihat source_manifest.csv.")
    conflicts = manifest.groupby("sha256_pixels").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Ada piksel identik dengan label bertentangan. Audit label dahulu.")

    original_train = manifest[manifest.source_split == "train"].copy()
    test = manifest[manifest.source_split == "test"].copy()
    test_hashes = set(test.sha256_pixels)
    overlap = original_train.sha256_pixels.isin(test_hashes)
    eligible_train = original_train.loc[~overlap].copy()
    unique_train = eligible_train.drop_duplicates("sha256_pixels").copy()
    train, validation = train_test_split(
        unique_train, test_size=config.validation_fraction, random_state=config.seed,
        stratify=unique_train.label,
    )
    splits = {"train": train.reset_index(drop=True),
              "validation": validation.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    hash_sets = {key: set(frame.sha256_pixels) for key, frame in splits.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if hash_sets[first] & hash_sets[second]:
            raise RuntimeError("Pemisahan data gagal: masih ada gambar identik antar split.")
    counts = {}
    for name, frame in splits.items():
        if set(frame.label) != {0, 1}:
            raise ValueError(f"Split {name} harus memiliki kedua kelas.")
        frame.to_csv(output_dir / f"{name}_manifest.csv", index=False)
        counts[name] = {label: int((frame.label == number).sum())
                        for label, number in LABELS.items()}
    audit = {
        "source_counts": manifest.groupby(["source_split", "class_name"]).size().to_dict(),
        "removed_train_test_overlap": int(overlap.sum()),
        "removed_repeated_train_pixels": len(eligible_train) - len(unique_train),
        "repeated_test_pixels_retained": int(test.sha256_pixels.duplicated().sum()),
        "split_counts": counts,
        "scope": "SHA256 piksel RGB identik; kemiripan visual dan gambar turunan belum diperiksa.",
        "test_policy": "Pertahankan seluruh test resmi; hapus overlap hanya dari train.",
    }
    audit["source_counts"] = {"/".join(key): int(value)
                              for key, value in audit["source_counts"].items()}
    save_json(output_dir / "data_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return splits, audit


def make_dataset(frame, root, config, training=False):
    """Input float32 0..255; rescaling berada di dalam model untuk konsistensi inferensi."""
    paths = [str(Path(root) / path) for path in frame.path]
    dataset = tf.data.Dataset.from_tensor_slices((paths, frame.label.to_numpy(dtype=np.float32)))
    if training:
        dataset = dataset.shuffle(len(frame), seed=config.seed, reshuffle_each_iteration=True)

    def read(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [config.image_size, config.image_size], method="bilinear")
        return image, tf.reshape(label, [1])

    return dataset.map(read, num_parallel_calls=tf.data.AUTOTUNE).batch(
        config.batch_size).prefetch(tf.data.AUTOTUNE)


# %% MODEL
def build_model(config):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(32, 32, 3), name="rgb_0_255"),
        layers.Rescaling(1.0 / 255, name="normalisasi"),
        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv_1"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_2"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv_3"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5, seed=config.seed),
        layers.Dense(1, activation="sigmoid", name="skor_fake"),
    ], name="CNN_CIFAKE")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="auc")])
    return model


def train_model(splits, root, config, output_dir):
    output_dir = Path(output_dir)
    train_ds = make_dataset(splits["train"], root, config, training=True)
    validation_ds = make_dataset(splits["validation"], root, config)
    model = build_model(config)
    model.summary()
    summary = []
    model.summary(print_fn=lambda line, **kwargs: summary.append(line))
    (output_dir / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output_dir / "best_model.keras"),
                                           monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]
    history = model.fit(train_ds, validation_data=validation_ds,
                        epochs=config.epochs, callbacks=callbacks, shuffle=False, verbose=2)
    # Evaluasi selalu memuat checkpoint terbaik menurut validation loss.
    best_model = tf.keras.models.load_model(output_dir / "best_model.keras")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(history.history[metric], label="Train")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch (mulai 0)", ylabel=metric, title=metric.capitalize())
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)
    return best_model, history


# %% EVALUATION
def evaluate_model(model, splits, root, config, output_dir):
    """Panggil setelah model dan threshold dibekukan; test tidak dipakai memilih model."""
    output_dir = Path(output_dir)
    frame = splits["test"].copy()
    scores = model.predict(make_dataset(frame, root, config), verbose=0).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Prediksi berisi nilai non-finite.")
    y_true = frame.label.to_numpy(dtype=int)
    y_pred = (scores >= config.threshold).astype(int)
    majority_label = int(splits["train"].label.mode().iloc[0])
    metrics = {
        "status": "evaluated", "positive_class": "FAKE", "class_mapping": LABELS,
        "threshold": config.threshold, "test_samples": len(frame),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "majority_baseline_accuracy": float(np.mean(y_true == majority_label)),
        "majority_baseline_class": "FAKE" if majority_label else "REAL",
        "confusion_matrix_labels": ["REAL", "FAKE"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    save_json(output_dir / "metrics.json", metrics)
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["REAL", "FAKE"], zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    frame["score_fake"] = scores
    frame["predicted_label"] = y_pred
    frame["predicted_class"] = np.where(y_pred == 1, "FAKE", "REAL")
    frame["correct"] = y_true == y_pred
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1],
                                           display_labels=["REAL", "FAKE"],
                                           ax=axis, cmap="Blues", colorbar=False)
    axis.set(title="Confusion matrix — test", xlabel="Prediksi", ylabel="Label sebenarnya")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, scores, ax=axis, name="FAKE = positif")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_title("ROC — test")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(figure)
    errors = frame.loc[~frame.correct].head(8)
    if len(errors):
        figure, axes = plt.subplots(2, 4, figsize=(10, 5))
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, errors.itertuples()):
            with Image.open(Path(root) / row.path) as image:
                axis.imshow(image.convert("RGB"), interpolation="nearest")
            axis.set_title(f"Asli: {row.class_name}\nPrediksi: {row.predicted_class}\n"
                           f"Skor FAKE: {row.score_fake:.3f}", fontsize=9)
        figure.suptitle("Maksimum 8 kesalahan pertama menurut urutan manifest")
        figure.tight_layout()
        figure.savefig(output_dir / "error_examples.png", dpi=160)
        plt.close(figure)
    print(report)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def predict_image(model, image_path, threshold=0.5):
    """Skor sigmoid bukan probabilitas keaslian yang sudah dikalibrasi."""
    raw = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    resized = tf.image.resize(raw, [32, 32], method="bilinear")
    score = float(model(resized[None, ...], training=False).numpy().reshape(-1)[0])
    return {"image": Path(image_path).name, "score_fake": score,
            "predicted_class": "FAKE" if score >= threshold else "REAL",
            "threshold": threshold,
            "note": "Prediksi pola CIFAKE; bukan bukti pasti asal-usul gambar."}


# %% RUN
def start_run(output_dir, config, data_source="CIFAKE dari penyimpanan lokal"):
    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Folder hasil harus kosong agar eksperimen tidak tercampur: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    versions = {}
    for package in ("tensorflow", "keras", "numpy", "pandas", "Pillow", "scikit-learn", "matplotlib", "kagglehub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    save_json(output_dir / "run_config.json", {
        "config": asdict(config), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source, "python": platform.python_version(),
        "package_versions": versions, "class_mapping": LABELS,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
    })
    return output_dir


def run_experiment(root, output_dir, config=None):
    config = config or Config()
    output_dir = start_run(output_dir, config)
    root = find_dataset_root(root)
    splits, audit = prepare_splits(root, config, output_dir)
    model, history = train_model(splits, root, config, output_dir)
    metrics = evaluate_model(model, splits, root, config, output_dir)
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Latih dan evaluasi CNN.")
    train.add_argument("--data-dir", type=Path, help="Folder CIFAKE; kosongkan untuk unduh otomatis.")
    train.add_argument("--output-dir", type=Path, default=Path("results/cifake_01"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict", help="Prediksi satu gambar menggunakan checkpoint.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        config = Config(epochs=arguments.epochs, batch_size=arguments.batch_size, seed=arguments.seed)
        config.validate()
        root = find_dataset_root(arguments.data_dir) if arguments.data_dir else download_dataset()
        run_experiment(root, arguments.output_dir, config)
    else:
        model = tf.keras.models.load_model(arguments.model, compile=False)
        print(json.dumps(predict_image(model, arguments.image), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
