import ast
import glob
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset


ELICE_PROJECT_DIR = "/home/elicer"
ELICE_DATASET_DIR = "/mnt/elice/dataset"

PROJECT_DIR = ELICE_PROJECT_DIR if os.path.isdir(ELICE_PROJECT_DIR) else (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
HISTORY_DIR = os.path.join(PROJECT_DIR, "history")
CACHE_DIR = os.environ.get("PERCH_FEATURE_CACHE_DIR", os.path.join(PROJECT_DIR, "perch_feature_cache"))
ARTIFACT_DIR = os.environ.get(
    "PERCH_KAGGLE_ARTIFACT_DIR",
    os.path.join(PROJECT_DIR, "kaggle_inference_artifacts"),
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(PROJECT_DIR, "tfhub_cache"))


def resolve_default_data_path():
    candidates = [
        os.path.join(ELICE_DATASET_DIR, "birdclef-2026"),
        ELICE_DATASET_DIR,
        os.path.join(PROJECT_DIR, "birdclef-2026"),
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "taxonomy.csv")):
            return path
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.join(PROJECT_DIR, "birdclef-2026")


DATA_PATH = os.environ.get("BIRDCLEF_DATA_PATH", resolve_default_data_path())
PERCH_MODEL_HANDLE = os.environ.get(
    "PERCH_MODEL_HANDLE",
    os.environ.get("BIRDCLEF_PERCH_MODEL_HANDLE", "https://tfhub.dev/google/bird-vocalization-classifier/4"),
)
DEFAULT_PERCH_ONNX_PATHS = [
    "/home/elicer/perch_v2_backbone.onnx",
    "/home/elicer/perch-onnx-backbone/perch_v2_backbone.onnx",
    "/home/elicer/teacher_soft_labels/perch_v2_backbone.onnx",
    "/mnt/elice/dataset/perch-onnx-backbone/perch_v2_backbone.onnx",
    "/mnt/elice/dataset/datasets/unsseo/perch-onnx-backbone/perch_v2_backbone.onnx",
    "/kaggle/input/datasets/unsseo/perch-onnx-backbone/perch_v2_backbone.onnx",
]
DEFAULT_PERCH_BACKEND = os.environ.get("PERCH_BACKEND", "onnx").strip().lower()


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def load_taxonomy_labels(data_path):
    taxonomy_path = os.path.join(data_path, "taxonomy.csv")
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"taxonomy.csv was not found under {data_path}")
    taxonomy = pd.read_csv(taxonomy_path)
    label_col = first_existing_column(taxonomy, ["primary_label", "ebird_code", "species_code", "label"])
    if label_col is None:
        raise ValueError("taxonomy.csv must contain a primary_label, ebird_code, species_code, or label column.")
    return taxonomy[label_col].dropna().astype(str).tolist()


def load_train_audio_labels(data_path):
    train_path = os.path.join(data_path, "train.csv")
    if not os.path.exists(train_path):
        return set()
    train = pd.read_csv(train_path)
    if "primary_label" not in train.columns:
        return set()
    return set(train["primary_label"].dropna().astype(str).tolist())


def parse_label_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        raw_items = None
        if text.startswith("[") or text.startswith("(") or text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    raw_items = list(parsed)
                else:
                    raw_items = [parsed]
            except Exception:
                raw_items = None
        if raw_items is None:
            raw_items = []
            for token in text.replace(",", " ").replace(";", " ").replace("|", " ").split():
                raw_items.append(token)

    labels = []
    for item in raw_items:
        label = str(item).strip().strip("'\"[](){}")
        if label and label.lower() not in {"nan", "none", "null", "nocall", "no_call", "background"}:
            labels.append(label)
    return labels


def find_soundscape_label_csv(data_path, explicit_path=None):
    env_path = explicit_path or os.environ.get("SOUNDSCAPE_WEAK_LABEL_CSV") or os.environ.get("TRAIN_SOUNDSCAPE_LABEL_CSV")
    if env_path:
        return env_path

    candidates = [
        "train_soundscape_labels.csv",
        "train_soundscapes_labels.csv",
        "train_soundscape.csv",
        "train_soundscapes.csv",
        "soundscape_labels.csv",
        "soundscapes_labels.csv",
    ]
    for name in candidates:
        path = os.path.join(data_path, name)
        if os.path.exists(path):
            return path

    patterns = [
        os.path.join(data_path, "*soundscape*label*.csv"),
        os.path.join(data_path, "*soundscape*.csv"),
    ]
    for pattern in patterns:
        matches = [
            path for path in sorted(glob.glob(pattern))
            if "sample" not in os.path.basename(path).lower()
            and "submission" not in os.path.basename(path).lower()
        ]
        if matches:
            return matches[0]
    return None


def parse_time_seconds(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) in {2, 3}:
        try:
            nums = [float(part) for part in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            minutes, seconds = nums
            return minutes * 60.0 + seconds
        hours, minutes, seconds = nums
        return hours * 3600.0 + minutes * 60.0 + seconds
    return None


def resolve_soundscape_audio_path(soundscape_dir, filename):
    filename = os.path.basename(str(filename))
    direct = os.path.join(soundscape_dir, filename)
    if os.path.exists(direct):
        return direct

    stem, ext = os.path.splitext(filename)
    candidates = [filename] if ext else []
    if not ext:
        for audio_ext in [".ogg", ".wav", ".flac", ".mp3"]:
            candidates.append(filename + audio_ext)

    for candidate in candidates:
        path = os.path.join(soundscape_dir, candidate)
        if os.path.exists(path):
            return path

    for path in glob.glob(os.path.join(soundscape_dir, "*")):
        if os.path.splitext(os.path.basename(path))[0] == stem:
            return path
    return None


def infer_filename_from_row(row, filename_col=None, row_id_col=None):
    if filename_col:
        return os.path.basename(str(row[filename_col]))
    if row_id_col:
        row_id = str(row[row_id_col])
        parts = row_id.split("_")
        if len(parts) >= 2 and parse_time_seconds(parts[-1]) is not None:
            return "_".join(parts[:-1])
        return row_id
    return None


def row_start_seconds(row, window_sec, start_col=None, end_col=None, seconds_col=None):
    if start_col:
        return parse_time_seconds(row[start_col])
    if seconds_col:
        seconds = parse_time_seconds(row[seconds_col])
        return None if seconds is None else max(0.0, seconds - float(window_sec))
    if end_col:
        seconds = parse_time_seconds(row[end_col])
        return None if seconds is None else max(0.0, seconds - float(window_sec))
    return None


def soundscape_duration_seconds(filepath):
    import torchaudio

    try:
        info = torchaudio.info(filepath)
        return info.num_frames / float(info.sample_rate)
    except Exception:
        wav, sr = torchaudio.load(filepath)
        return wav.shape[-1] / float(sr)


def labels_from_soundscape_row(row, labels, label_col=None):
    label_set = set(labels)
    if label_col:
        return [label for label in parse_label_list(row[label_col]) if label in label_set]

    parsed = []
    for candidate in [
        "primary_label",
        "birds",
        "labels",
        "label",
        "target",
        "targets",
        "species",
        "ebird_code",
        "secondary_labels",
    ]:
        if candidate in row.index:
            parsed.extend(parse_label_list(row[candidate]))
    parsed = [label for label in parsed if label in label_set]
    if parsed:
        return sorted(set(parsed))

    one_hot = []
    for label in labels:
        if label in row.index:
            try:
                if float(row[label]) > 0:
                    one_hot.append(label)
            except Exception:
                pass
    return one_hot


def limit_rows_with_label_coverage(rows, max_rows, min_per_label=1, seed=42):
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    rng = np.random.default_rng(seed)
    keep = set()
    label_to_indices = {}
    for idx, labels in rows["target_labels"].items():
        for label in labels:
            label_to_indices.setdefault(label, []).append(idx)

    for indices in label_to_indices.values():
        take = min(max(1, min_per_label), len(indices))
        keep.update(rng.choice(np.array(indices), size=take, replace=False).tolist())

    if len(keep) > max_rows:
        keep = set(rng.choice(np.array(sorted(keep)), size=max_rows, replace=False).tolist())
    elif len(keep) < max_rows:
        rest = np.array([idx for idx in rows.index if idx not in keep])
        if len(rest) > 0:
            extra = rng.choice(rest, size=min(max_rows - len(keep), len(rest)), replace=False).tolist()
            keep.update(extra)

    limited = rows.loc[sorted(keep)].reset_index(drop=True)
    before_labels = set(label for labels in rows["target_labels"] for label in labels)
    after_labels = set(label for labels in limited["target_labels"] for label in labels)
    print(
        "row limit with label coverage:",
        len(rows),
        "->",
        len(limited),
        "covered labels:",
        len(after_labels),
        "/",
        len(before_labels),
    )
    missing = sorted(before_labels - after_labels)
    if missing:
        print("labels lost by row limit:", missing[:20])
    return limited


def build_soundscape_rows(data_path, labels, cfg):
    soundscape_dir = cfg["audio_dir"]
    label_csv = find_soundscape_label_csv(data_path, cfg.get("label_csv"))
    if label_csv is None or not os.path.exists(label_csv):
        raise FileNotFoundError(
            "No train_soundscapes label CSV was found. Set SOUNDSCAPE_WEAK_LABEL_CSV if it is not in DATA_PATH."
        )

    weak_df = pd.read_csv(label_csv)
    filename_col = cfg.get("filename_col") or first_existing_column(
        weak_df,
        [
            "filename",
            "file_name",
            "audio_filename",
            "audio",
            "audio_id",
            "soundscape_id",
            "recording_id",
            "filepath",
            "path",
            "soundscape_filename",
            "soundscape",
        ],
    )
    row_id_col = cfg.get("row_id_col") or first_existing_column(weak_df, ["row_id", "id"])
    label_col = cfg.get("label_col")
    start_col = cfg.get("start_col") or first_existing_column(
        weak_df, ["start_seconds", "start_second", "start", "begin", "begin_seconds"]
    )
    end_col = cfg.get("end_col") or first_existing_column(
        weak_df, ["end_seconds", "end_second", "end", "stop", "seconds_end"]
    )
    seconds_col = cfg.get("seconds_col") or first_existing_column(weak_df, ["seconds", "second"])
    window_sec = float(cfg.get("window_sec", 5.0))
    stride_sec = float(cfg.get("stride_sec", window_sec))
    file_level_policy = cfg.get("file_level_policy", "expand")

    rows = []
    missing_files = 0
    for _, row in weak_df.iterrows():
        row_labels = labels_from_soundscape_row(row, labels, label_col=label_col)
        if not row_labels:
            continue

        filename = infer_filename_from_row(row, filename_col=filename_col, row_id_col=row_id_col)
        if filename is None:
            continue
        filepath = resolve_soundscape_audio_path(soundscape_dir, filename)
        if filepath is None:
            missing_files += 1
            continue

        filename = os.path.basename(filepath)
        start = row_start_seconds(row, window_sec, start_col=start_col, end_col=end_col, seconds_col=seconds_col)
        if start is not None:
            starts = [round(float(start), 3)]
        elif file_level_policy == "first":
            starts = [0.0]
        else:
            duration = soundscape_duration_seconds(filepath)
            max_start = max(0.0, duration - window_sec)
            starts = np.arange(0.0, max_start + 1e-6, stride_sec).round(3).tolist()
            if not starts:
                starts = [0.0]

        for start in starts:
            rows.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "start_seconds": float(start),
                    "target_labels": tuple(sorted(set(row_labels))),
                }
            )

    if not rows:
        raise ValueError(f"No usable soundscape weak-label rows were built from {label_csv}.")

    rows = pd.DataFrame(rows)
    rows = rows.groupby(["filename", "filepath", "start_seconds"], as_index=False).agg(
        {"target_labels": lambda values: tuple(sorted(set(label for labels_ in values for label in labels_)))}
    )
    rows = limit_rows_with_label_coverage(
        rows,
        max_rows=int(cfg.get("max_rows", 0) or 0),
        min_per_label=int(cfg.get("min_per_label", 1)),
        seed=int(cfg.get("seed", 42)),
    )

    print(
        "soundscape labels:",
        os.path.basename(label_csv),
        "rows:",
        len(rows),
        "missing files:",
        missing_files,
    )
    return rows.reset_index(drop=True)


def make_targets(rows, labels):
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for row_idx, row_labels in enumerate(rows["target_labels"]):
        for label in row_labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                y[row_idx, idx] = 1.0
    return y


def load_audio_window(filepath, start_seconds, sample_rate=32000, window_sec=5.0):
    import torchaudio

    wav, sr = torchaudio.load(filepath)
    wav = wav.float().mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    target_len = int(round(sample_rate * window_sec))
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    segment = wav[start : start + target_len]
    if segment.numel() < target_len:
        padded = torch.zeros(target_len, dtype=wav.dtype)
        padded[: segment.numel()] = segment
        segment = padded
    return segment.detach().cpu().numpy().astype("float32")


def configure_tensorflow():
    import tensorflow as tf

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    return tf


def load_perch_model():
    tf = configure_tensorflow()
    model_dir = os.environ.get("PERCH_MODEL_DIR")
    if model_dir and os.path.exists(model_dir):
        print("Loading Perch SavedModel:", model_dir)
        return tf.saved_model.load(model_dir)

    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise ImportError(
            "tensorflow_hub is required to download Perch from TFHub. "
            "Install it or set PERCH_MODEL_DIR to a local SavedModel."
        ) from exc

    print("Loading Perch from TFHub:", PERCH_MODEL_HANDLE)
    return hub.load(PERCH_MODEL_HANDLE)


def find_perch_onnx_path():
    explicit_path = os.environ.get("PERCH_ONNX_PATH")
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"PERCH_ONNX_PATH does not exist: {explicit_path}")
        return explicit_path

    candidates = list(DEFAULT_PERCH_ONNX_PATHS)
    for root in [PROJECT_DIR, ELICE_DATASET_DIR, "/kaggle/input"]:
        if root and os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "perch_v2_backbone.onnx"), recursive=True))
            candidates.extend(glob.glob(os.path.join(root, "**", "*perch*backbone*.onnx"), recursive=True))

    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "perch_v2_backbone.onnx was not found. Set PERCH_ONNX_PATH to the ONNX backbone file."
    )


def load_perch_engine(cfg=None):
    backend = str((cfg or {}).get("backend") or DEFAULT_PERCH_BACKEND).strip().lower()
    if backend in {"onnx", "ort", "onnxruntime"}:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required because PERCH_BACKEND=onnx. "
                "Install onnxruntime before running this training script."
            ) from exc
        onnx_path = find_perch_onnx_path()
        available_providers = ort.get_available_providers()
        providers = []
        if torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        print("Loading Perch ONNX backbone:", onnx_path)
        print("ONNX providers:", providers)
        session = ort.InferenceSession(onnx_path, providers=providers)
        print("Perch ONNX inputs:", [(inp.name, inp.shape, inp.type) for inp in session.get_inputs()])
        print("Perch ONNX outputs:", [(out.name, out.shape, out.type) for out in session.get_outputs()])
        return "onnx", session, onnx_path

    if backend in {"tf", "tfhub", "savedmodel"}:
        return "tf", load_perch_model(), os.environ.get("PERCH_MODEL_DIR") or PERCH_MODEL_HANDLE

    raise ValueError("PERCH_BACKEND must be onnx or tf.")


def tensor_to_numpy(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def dict_get_first(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def infer_perch_batch(model, audio_batch):
    outputs = model.infer_tf(audio_batch.astype("float32"))
    if isinstance(outputs, dict):
        logits = dict_get_first(outputs, ["logits", "label", "scores", "class_logits"])
        embedding = dict_get_first(outputs, ["embedding", "embeddings", "embed", "feature", "features"])
    elif isinstance(outputs, (tuple, list)):
        if len(outputs) < 2:
            raise ValueError("Perch infer_tf returned fewer than two outputs; expected logits and embedding.")
        logits, embedding = outputs[0], outputs[1]
    else:
        raise ValueError(f"Unsupported Perch infer_tf output type: {type(outputs)}")

    if logits is None or embedding is None:
        raise ValueError("Could not read logits and embedding from Perch infer_tf output.")

    logits = tensor_to_numpy(logits).astype("float32")
    embedding = tensor_to_numpy(embedding).astype("float32")
    if logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)
    if embedding.ndim > 2:
        embedding = embedding.reshape(embedding.shape[0], -1)
    return logits, embedding


def infer_perch_onnx_batch(session, audio_batch):
    audio_batch = audio_batch.astype("float32")
    input_name = os.environ.get("PERCH_ONNX_INPUT_KEY") or session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(None, {input_name: audio_batch})
    output_map = {name: value for name, value in zip(output_names, outputs)}

    embedding_key = os.environ.get("PERCH_ONNX_EMBEDDING_KEY", "embedding")
    if embedding_key in output_map:
        embedding = output_map[embedding_key]
    elif len(outputs) == 1:
        embedding = outputs[0]
    else:
        embedding = max(outputs, key=lambda value: int(np.asarray(value).reshape(value.shape[0], -1).shape[1]))

    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim > 2:
        embedding = embedding.reshape(embedding.shape[0], -1)
    if embedding.ndim != 2 or embedding.shape[0] != audio_batch.shape[0]:
        raise ValueError(
            f"Perch ONNX embedding output must be batch-aligned 2D, got {embedding.shape}."
        )

    logits = np.zeros((audio_batch.shape[0], 0), dtype=np.float32)
    return logits, embedding


def default_cache_path(cfg):
    backend = str(cfg.get("backend") or DEFAULT_PERCH_BACKEND).strip().lower()
    name = f"soundscape_perch_{backend}_features_{int(float(cfg['window_sec']) * 1000)}ms"
    return os.path.join(CACHE_DIR, name + ".npz")


def extract_or_load_perch_features(rows, labels, targets, cfg):
    cache_path = cfg.get("cache_path") or default_cache_path(cfg)
    force = bool(cfg.get("force_recreate", False))
    feature_mode = cfg.get("feature_mode", "emb").lower()
    backend = str(cfg.get("backend") or DEFAULT_PERCH_BACKEND).strip().lower()
    expected_emb_dim = int(cfg.get("expected_embedding_dim", 1536 if backend in {"onnx", "ort", "onnxruntime"} else 0))

    if os.path.exists(cache_path) and not force:
        print("Loading cached Perch features:", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        cached_labels = data["labels"].astype(str).tolist()
        if cached_labels != list(labels):
            raise ValueError("Cached labels do not match taxonomy labels. Set PERCH_FEATURE_FORCE_RECREATE=1.")
        cached_backend = str(data["backend"].item()) if "backend" in data.files else None
        if cached_backend and cached_backend != backend:
            raise ValueError(
                f"Cached backend is {cached_backend}, but current PERCH_BACKEND is {backend}. "
                "Set PERCH_FEATURE_FORCE_RECREATE=1 or use a different PERCH_FEATURE_CACHE_PATH."
            )
        result = {
            "emb": data["emb"].astype("float32"),
            "logits": data["logits"].astype("float32") if "logits" in data.files else None,
            "y": data["y"].astype("float32"),
            "filenames": data["filenames"].astype(str),
            "start_seconds": data["start_seconds"].astype("float32"),
            "labels": cached_labels,
        }
        if "feature_source" in data.files:
            cfg["feature_source"] = str(data["feature_source"].item())
        cfg["backend"] = cached_backend or backend
        if feature_mode in {"logits", "emb_logits", "concat"} and result["logits"] is None:
            raise ValueError(
                "The cache does not contain logits. Set PERCH_FEATURE_FORCE_RECREATE=1 "
                "or use PERCH_FEATURE_MODE=emb."
            )
        if expected_emb_dim and result["emb"].shape[1] != expected_emb_dim:
            raise ValueError(
                f"Cached embedding dim is {result['emb'].shape[1]}, but expected {expected_emb_dim}. "
                "Set PERCH_FEATURE_FORCE_RECREATE=1 to recreate features with perch_v2_backbone.onnx."
            )
        return result

    engine_kind, model, feature_source = load_perch_engine(cfg)
    cfg["backend"] = engine_kind
    cfg["feature_source"] = str(feature_source)
    batch_size = int(cfg.get("batch_size", 16))
    sample_rate = int(cfg.get("sample_rate", 32000))
    window_sec = float(cfg.get("window_sec", 5.0))
    save_logits = bool(cfg.get("save_logits", False)) or feature_mode in {"logits", "emb_logits", "concat"}

    embeddings = []
    logits_list = []
    t0 = time.time()
    n = len(rows)
    for start_idx in range(0, n, batch_size):
        batch_rows = rows.iloc[start_idx : start_idx + batch_size]
        audio = np.stack(
            [
                load_audio_window(row.filepath, row.start_seconds, sample_rate=sample_rate, window_sec=window_sec)
                for row in batch_rows.itertuples(index=False)
            ]
        )
        if engine_kind == "onnx":
            logits, emb = infer_perch_onnx_batch(model, audio)
        else:
            logits, emb = infer_perch_batch(model, audio)
        embeddings.append(emb)
        if save_logits:
            logits_list.append(logits)

        done = min(start_idx + batch_size, n)
        elapsed = max(1e-6, time.time() - t0)
        rate = done / elapsed
        remaining = (n - done) / max(rate, 1e-6)
        print(
            f"perch features {done}/{n} ({done / n * 100:.1f}%) "
            f"elapsed {elapsed / 60:.1f}m eta {remaining / 60:.1f}m"
        )

    emb = np.concatenate(embeddings, axis=0).astype("float32")
    logits = np.concatenate(logits_list, axis=0).astype("float32") if save_logits else None

    save_kwargs = {
        "emb": emb,
        "y": targets.astype("float32"),
        "filenames": rows["filename"].astype(str).to_numpy(),
        "start_seconds": rows["start_seconds"].astype("float32").to_numpy(),
        "labels": np.asarray(labels, dtype=object),
        "backend": np.asarray(engine_kind, dtype=object),
        "feature_source": np.asarray(str(feature_source), dtype=object),
    }
    if logits is not None:
        save_kwargs["logits"] = logits

    np.savez_compressed(cache_path, **save_kwargs)
    rows_path = cache_path.replace(".npz", "_rows.csv")
    meta_path = cache_path.replace(".npz", "_meta.json")
    rows.to_csv(rows_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": DATA_PATH,
                "feature_mode": feature_mode,
                "window_sec": window_sec,
                "sample_rate": sample_rate,
                "rows": int(len(rows)),
                "embedding_dim": int(emb.shape[1]),
                "logit_dim": int(logits.shape[1]) if logits is not None else 0,
                "backend": engine_kind,
                "feature_source": str(feature_source),
            },
            f,
            indent=2,
        )
    print("Saved Perch feature cache:", cache_path)

    return {
        "emb": emb,
        "logits": logits,
        "y": targets.astype("float32"),
        "filenames": rows["filename"].astype(str).to_numpy(),
        "start_seconds": rows["start_seconds"].astype("float32").to_numpy(),
        "labels": list(labels),
    }


def build_feature_matrix(feature_data, feature_mode):
    feature_mode = feature_mode.lower()
    emb = feature_data["emb"]
    logits = feature_data.get("logits")
    if feature_mode == "emb":
        x = emb
    elif feature_mode == "logits":
        if logits is None:
            raise ValueError("PERCH_FEATURE_MODE=logits requires cached or extracted logits.")
        x = logits
    elif feature_mode in {"emb_logits", "concat"}:
        if logits is None:
            raise ValueError("PERCH_FEATURE_MODE=emb_logits requires cached or extracted logits.")
        x = np.concatenate([emb, logits], axis=1)
    else:
        raise ValueError("PERCH_FEATURE_MODE must be emb, logits, or emb_logits.")
    return np.nan_to_num(x.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)


class PerchProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_dims(text):
    dims = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    return dims or [512, 256]


def macro_auc(y_true, y_score, indices=None):
    if indices is None:
        indices = range(y_true.shape[1])
    aucs = []
    valid_indices = []
    for idx in indices:
        target = y_true[:, idx]
        if target.min() == target.max():
            continue
        try:
            auc = roc_auc_score(target, y_score[:, idx])
        except ValueError:
            continue
        if not math.isnan(auc):
            aucs.append(float(auc))
            valid_indices.append(idx)
    if not aucs:
        return float("nan"), 0, []
    return float(np.mean(aucs)), len(aucs), valid_indices


def evaluate_predictions(y_true, y_score, labels, zero_shot_labels):
    zero_set = set(zero_shot_labels)
    zero_indices = [idx for idx, label in enumerate(labels) if label in zero_set]
    seen_indices = [idx for idx, label in enumerate(labels) if label not in zero_set]
    all_auc, all_valid, _ = macro_auc(y_true, y_score)
    zero_auc, zero_valid, _ = macro_auc(y_true, y_score, zero_indices)
    seen_auc, seen_valid, _ = macro_auc(y_true, y_score, seen_indices)
    return {
        "macro_auc": all_auc,
        "valid_auc_labels": all_valid,
        "zero_shot_auc": zero_auc,
        "valid_zero_shot_labels": zero_valid,
        "seen_auc": seen_auc,
        "valid_seen_labels": seen_valid,
    }


def standardize_fold(x, train_idx, val_idx):
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x[train_idx] - mean) / std, (x[val_idx] - mean) / std, mean.astype("float32"), std.astype("float32")


def make_splits(y, groups, folds, seed, run_all_folds, selected_fold):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        n_splits = min(int(folds), len(unique_groups))
        n_splits = max(2, n_splits)
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups=groups))
    else:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        splits = [(train_idx, val_idx)]

    if run_all_folds:
        return splits
    selected_fold = int(selected_fold) % len(splits)
    return [splits[selected_fold]]


def train_probe_fold(x, y, labels, zero_shot_labels, train_idx, val_idx, fold, cfg):
    device = torch.device(cfg["device"])
    x_train, x_val, mean, std = standardize_fold(x, train_idx, val_idx)
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = PerchProbe(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dims=parse_hidden_dims(cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = np.ones_like(pos, dtype=np.float32)
    positive_mask = pos > 0
    pos_weight[positive_mask] = neg[positive_mask] / np.maximum(pos[positive_mask], 1.0)
    pos_weight = np.clip(pos_weight, 1.0, float(cfg["max_pos_weight"]))

    label_weight = np.ones(y.shape[1], dtype=np.float32)
    zero_set = set(zero_shot_labels)
    zero_indices = [idx for idx, label in enumerate(labels) if label in zero_set]
    if zero_indices:
        label_weight[zero_indices] = float(cfg["zero_shot_loss_weight"])

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
    )
    label_weight_t = torch.tensor(label_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train_batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    best_pred = None
    patience = int(cfg["patience"])
    log_every = max(1, int(cfg["log_every"]))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_matrix = criterion(logits, yb) * label_weight_t
            loss = loss_matrix.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            total_count += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(x_val, dtype=torch.float32, device=device))
            val_pred = torch.sigmoid(val_logits).detach().cpu().numpy().astype("float32")
        metrics = evaluate_predictions(y_val, val_pred, labels, zero_shot_labels)
        metric = metrics["zero_shot_auc"]
        if math.isnan(metric):
            metric = metrics["macro_auc"]

        improved = not math.isnan(metric) and metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_pred = val_pred.copy()
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "mean": mean,
                "std": std,
                "labels": list(labels),
                "zero_shot_labels": list(zero_shot_labels),
                "cfg": dict(cfg),
                "fold": int(fold),
                "best_epoch": int(best_epoch),
                "metrics": dict(metrics),
            }

        if epoch == 1 or epoch % log_every == 0 or improved:
            train_loss = total_loss / max(1, total_count)
            print(
                f"fold {fold} epoch {epoch:03d} loss {train_loss:.5f} "
                f"macro_auc {metrics['macro_auc']:.5f} zero_auc {metrics['zero_shot_auc']:.5f}"
            )

        if epoch - best_epoch >= patience:
            print(f"fold {fold} early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    if best_pred is None:
        model.eval()
        with torch.no_grad():
            best_pred = torch.sigmoid(model(torch.tensor(x_val, dtype=torch.float32, device=device))).cpu().numpy()
        best_state = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "mean": mean,
            "std": std,
            "labels": list(labels),
            "zero_shot_labels": list(zero_shot_labels),
            "cfg": dict(cfg),
            "fold": int(fold),
            "best_epoch": int(best_epoch),
            "metrics": evaluate_predictions(y_val, best_pred, labels, zero_shot_labels),
        }

    model_path = os.path.join(MODELS_DIR, f"perch_feature_probe_fold{fold}.pt")
    torch.save(best_state, model_path)
    print("saved model:", model_path)
    metrics = dict(best_state["metrics"])
    metrics["best_epoch"] = int(best_state.get("best_epoch", 0))
    return best_pred, val_idx, metrics, model_path


def train_probe_full(x, y, labels, zero_shot_labels, cfg, full_epochs):
    device = torch.device(cfg["device"])
    mean = x.mean(axis=0, keepdims=True).astype("float32")
    std = x.std(axis=0, keepdims=True).astype("float32")
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    x_train = ((x - mean) / std).astype("float32")

    full_cfg = dict(cfg)
    full_cfg["epochs"] = int(full_epochs)
    model = PerchProbe(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dims=parse_hidden_dims(full_cfg["hidden_dims"]),
        dropout=float(full_cfg["dropout"]),
    ).to(device)

    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = np.ones_like(pos, dtype=np.float32)
    positive_mask = pos > 0
    pos_weight[positive_mask] = neg[positive_mask] / np.maximum(pos[positive_mask], 1.0)
    pos_weight = np.clip(pos_weight, 1.0, float(full_cfg["max_pos_weight"]))

    label_weight = np.ones(y.shape[1], dtype=np.float32)
    zero_set = set(zero_shot_labels)
    zero_indices = [idx for idx, label in enumerate(labels) if label in zero_set]
    if zero_indices:
        label_weight[zero_indices] = float(full_cfg["zero_shot_loss_weight"])

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
    )
    label_weight_t = torch.tensor(label_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(full_cfg["lr"]),
        weight_decay=float(full_cfg["weight_decay"]),
    )
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(full_cfg["train_batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    log_every = max(1, int(full_cfg["log_every"]))
    print("training full zero-shot probe on all weak-label rows; epochs:", full_epochs)
    for epoch in range(1, int(full_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (criterion(logits, yb) * label_weight_t).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            total_count += xb.shape[0]
        if epoch == 1 or epoch % log_every == 0 or epoch == int(full_epochs):
            print(f"full probe epoch {epoch:03d} loss {total_loss / max(1, total_count):.5f}")

    state = {
        "artifact_version": 1,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": mean,
        "std": std,
        "labels": list(labels),
        "zero_shot_labels": list(zero_shot_labels),
        "cfg": full_cfg,
        "fold": "full",
        "best_epoch": int(full_epochs),
        "metrics": {
            "train_rows": int(len(y)),
            "input_dim": int(x.shape[1]),
            "output_dim": int(y.shape[1]),
        },
    }
    model_path = os.path.join(MODELS_DIR, "perch_feature_probe_full.pt")
    torch.save(state, model_path)
    print("saved full probe model:", model_path)
    return model_path


def export_probe_artifacts(
    full_model_path,
    metrics_path,
    oof_prediction_path,
    labels,
    zero_shot_labels,
    feature_cfg,
    train_cfg,
    final_metrics,
    full_epochs,
    rows_count,
    input_dim,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stable_model_path = os.path.join(ARTIFACT_DIR, "perch_feature_probe_full.pt")
    stable_history_path = os.path.join(ARTIFACT_DIR, "perch_feature_probe_history.csv")
    stable_manifest_path = os.path.join(ARTIFACT_DIR, "perch_feature_probe_manifest.json")
    stable_labels_path = os.path.join(ARTIFACT_DIR, "perch_feature_probe_labels.csv")

    shutil.copy2(full_model_path, stable_model_path)
    if metrics_path and os.path.exists(metrics_path):
        shutil.copy2(metrics_path, os.path.join(ARTIFACT_DIR, "perch_feature_zeroshot_metrics.csv"))
    if oof_prediction_path and os.path.exists(oof_prediction_path):
        shutil.copy2(oof_prediction_path, os.path.join(ARTIFACT_DIR, "perch_feature_zeroshot_oof.csv"))

    history_row = {
        "artifact_version": 1,
        "created_at": timestamp,
        "model_file": "perch_feature_probe_full.pt",
        "feature_backend": feature_cfg.get("backend", DEFAULT_PERCH_BACKEND),
        "feature_source": feature_cfg.get("feature_source") or os.environ.get("PERCH_ONNX_PATH") or "perch_v2_backbone.onnx",
        "feature_mode": feature_cfg.get("feature_mode", "emb"),
        "window_sec": float(feature_cfg.get("window_sec", 5.0)),
        "sample_rate": int(feature_cfg.get("sample_rate", 32000)),
        "expected_embedding_dim": int(feature_cfg.get("expected_embedding_dim", input_dim)),
        "input_dim": int(input_dim),
        "output_dim": int(len(labels)),
        "train_rows": int(rows_count),
        "zero_shot_labels": len(zero_shot_labels),
        "full_epochs": int(full_epochs),
        "hidden_dims": train_cfg.get("hidden_dims"),
        "dropout": float(train_cfg.get("dropout", 0.0)),
        "lr": float(train_cfg.get("lr", 0.0)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "zero_shot_loss_weight": float(train_cfg.get("zero_shot_loss_weight", 1.0)),
        "oof_macro_auc": final_metrics.get("macro_auc"),
        "oof_zero_shot_auc": final_metrics.get("zero_shot_auc"),
        "valid_zero_shot_labels": final_metrics.get("valid_zero_shot_labels"),
    }
    pd.DataFrame([history_row]).to_csv(stable_history_path, index=False)
    pd.DataFrame({
        "label": list(labels),
        "is_zero_shot": [label in set(zero_shot_labels) for label in labels],
    }).to_csv(stable_labels_path, index=False)

    manifest = {
        **history_row,
        "labels": list(labels),
        "zero_shot_labels_list": list(zero_shot_labels),
        "artifact_dir": ARTIFACT_DIR,
        "required_for_kaggle": [
            "perch_feature_probe_full.pt",
            "perch_feature_probe_history.csv",
            "perch_feature_probe_manifest.json",
        ],
    }
    with open(stable_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("exported Kaggle probe artifacts:", ARTIFACT_DIR)
    print(" -", stable_model_path)
    print(" -", stable_history_path)
    print(" -", stable_manifest_path)
    return {
        "artifact_dir": ARTIFACT_DIR,
        "model": stable_model_path,
        "history": stable_history_path,
        "manifest": stable_manifest_path,
    }


def save_prediction_csv(path, filenames, starts, labels, predictions):
    pred_df = pd.DataFrame(predictions, columns=labels)
    pred_df.insert(0, "start_seconds", starts)
    pred_df.insert(0, "filename", filenames)
    pred_df.to_csv(path, index=False)
    print("saved predictions:", path)


def load_torch_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def default_prediction_audio_dir(data_path):
    explicit = (
        os.environ.get("HYBRID_PREDICTION_AUDIO_DIR")
        or os.environ.get("PERCH_PREDICTION_AUDIO_DIR")
        or os.environ.get("SOUNDSCAPE_PREDICTION_AUDIO_DIR")
    )
    if explicit:
        return explicit

    for name in ["test_soundscapes", "train_soundscapes"]:
        path = os.path.join(data_path, name)
        if os.path.isdir(path):
            return path
    return os.path.join(data_path, "test_soundscapes")


def parse_prediction_row_id(row_id, window_sec, time_mode="end"):
    text = str(row_id)
    stem, sep, sec_text = text.rpartition("_")
    if not sep:
        return text, None
    seconds = parse_time_seconds(sec_text)
    if seconds is None:
        return text, None
    if time_mode == "start":
        start = seconds
    else:
        start = max(0.0, seconds - float(window_sec))
    return stem, float(start)


def build_prediction_rows_from_csv(csv_path, data_path, cfg):
    pred_df = pd.read_csv(csv_path)
    audio_dir = cfg.get("audio_dir") or default_prediction_audio_dir(data_path)
    window_sec = float(cfg.get("window_sec", 5.0))
    row_id_time_mode = cfg.get("row_id_time_mode", "end")

    filename_col = cfg.get("filename_col") or first_existing_column(
        pred_df,
        [
            "filename",
            "file_name",
            "audio_filename",
            "audio",
            "audio_id",
            "soundscape_id",
            "recording_id",
            "filepath",
            "path",
            "soundscape_filename",
            "soundscape",
        ],
    )
    row_id_col = cfg.get("row_id_col") or first_existing_column(pred_df, ["row_id", "id"])
    start_col = cfg.get("start_col") or first_existing_column(
        pred_df, ["start_seconds", "start_second", "start", "begin", "begin_seconds"]
    )
    end_col = cfg.get("end_col") or first_existing_column(
        pred_df, ["end_seconds", "end_second", "end", "stop", "seconds_end"]
    )
    seconds_col = cfg.get("seconds_col") or first_existing_column(pred_df, ["seconds", "second"])

    rows = []
    missing_files = []
    for row_pos, row in pred_df.iterrows():
        filename = None
        start = None
        if filename_col:
            filename = os.path.basename(str(row[filename_col]))
        elif row_id_col:
            filename, start = parse_prediction_row_id(row[row_id_col], window_sec, time_mode=row_id_time_mode)

        if start_col:
            start = parse_time_seconds(row[start_col])
        elif seconds_col:
            seconds = parse_time_seconds(row[seconds_col])
            if seconds is not None:
                start = max(0.0, seconds - window_sec)
        elif end_col:
            seconds = parse_time_seconds(row[end_col])
            if seconds is not None:
                start = max(0.0, seconds - window_sec)

        if filename is None:
            raise ValueError(
                "Could not infer audio filename from baseline prediction CSV. "
                "Set HYBRID_FILENAME_COL or HYBRID_ROW_ID_COL."
            )
        if start is None:
            start = 0.0

        filepath = resolve_soundscape_audio_path(audio_dir, filename)
        if filepath is None:
            missing_files.append(filename)
            continue

        rows.append(
            {
                "row_position": int(row_pos),
                "filename": os.path.basename(filepath),
                "filepath": filepath,
                "start_seconds": round(float(start), 3),
            }
        )

    if missing_files:
        examples = missing_files[:5]
        raise FileNotFoundError(
            f"Missing audio files for {len(missing_files)} prediction rows under {audio_dir}, e.g. {examples}"
        )
    if not rows:
        raise ValueError(f"No prediction rows were built from {csv_path}")

    rows = pd.DataFrame(rows)
    print("hybrid prediction rows:", len(rows), "audio_dir:", audio_dir)
    return pred_df, rows


def predict_with_probe_models(x, model_paths, labels):
    if not model_paths:
        raise ValueError("No Perch probe model paths were provided.")

    preds = []
    for model_path in model_paths:
        state = load_torch_state(model_path)
        state_labels = [str(label) for label in state.get("labels", labels)]
        if state_labels != list(labels):
            raise ValueError(
                f"Probe model labels do not match current taxonomy labels: {model_path}. "
                "Retrain the probe or use the same taxonomy.csv."
            )

        cfg = state.get("cfg", {})
        device = torch.device(os.environ.get("PERCH_PROBE_INFER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        x_norm = (x - mean) / std

        model = PerchProbe(
            input_dim=x.shape[1],
            output_dim=len(labels),
            hidden_dims=parse_hidden_dims(cfg.get("hidden_dims", os.environ.get("PERCH_PROBE_HIDDEN_DIMS", "512,256"))),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        fold_preds = []
        batch_size = int(os.environ.get("PERCH_PROBE_INFER_BATCH_SIZE", "512"))
        with torch.no_grad():
            for start_idx in range(0, len(x_norm), batch_size):
                xb = torch.tensor(x_norm[start_idx : start_idx + batch_size], dtype=torch.float32, device=device)
                fold_preds.append(torch.sigmoid(model(xb)).detach().cpu().numpy().astype("float32"))
        preds.append(np.concatenate(fold_preds, axis=0))
        print("loaded probe model:", model_path)

    return np.mean(np.stack(preds, axis=0), axis=0).astype("float32")


def normalize_start_ms(values):
    return (pd.Series(values).astype(float) * 1000).round().astype("int64")


def replacement_key(df, window_sec=5.0):
    if "row_id" in df.columns:
        return df["row_id"].astype(str)
    if "filename" in df.columns and "start_seconds" in df.columns:
        stem = df["filename"].astype(str).map(lambda value: os.path.splitext(os.path.basename(value))[0])
        start_ms = normalize_start_ms(df["start_seconds"]).astype(str)
        return stem + "__" + start_ms
    if "filename" in df.columns and "seconds" in df.columns:
        stem = df["filename"].astype(str).map(lambda value: os.path.splitext(os.path.basename(value))[0])
        start_ms = normalize_start_ms(pd.Series(df["seconds"]).astype(float) - float(window_sec)).clip(lower=0).astype(str)
        return stem + "__" + start_ms
    return None


def replace_zero_shot_columns(
    baseline_df,
    probe_df,
    labels,
    zero_shot_labels,
    output_csv,
    alpha=1.0,
    positional=False,
    window_sec=5.0,
):
    out = baseline_df.copy()
    replace_labels = [label for label in zero_shot_labels if label in out.columns and label in probe_df.columns]
    if not replace_labels:
        raise ValueError("No zero-shot labels were present in both baseline and probe prediction CSVs.")

    alpha = float(alpha)
    if positional:
        if len(out) != len(probe_df):
            raise ValueError(
                "Positional replacement requires baseline and probe predictions to have the same number of rows."
            )
        for label in replace_labels:
            probe_values = probe_df[label].astype(float).to_numpy()
            base_values = out[label].astype(float).to_numpy()
            out[label] = (1.0 - alpha) * base_values + alpha * probe_values
    else:
        base_key = replacement_key(out, window_sec=window_sec)
        probe_key = replacement_key(probe_df, window_sec=window_sec)
        if base_key is None or probe_key is None:
            if len(out) == len(probe_df):
                print("No shared key columns found; falling back to row-order replacement.")
                return replace_zero_shot_columns(
                    baseline_df,
                    probe_df,
                    labels,
                    zero_shot_labels,
                    output_csv,
                    alpha=alpha,
                    positional=True,
                    window_sec=window_sec,
                )
            raise ValueError(
                "Could not align baseline and probe predictions. Include row_id or filename/start_seconds columns."
            )

        probe_work = probe_df.copy()
        probe_work["__key"] = probe_key.values
        probe_work = probe_work.drop_duplicates("__key", keep="last").set_index("__key")

        missing = []
        for row_idx, key in base_key.items():
            if key not in probe_work.index:
                missing.append(key)
                continue
            for label in replace_labels:
                base_value = float(out.at[row_idx, label])
                probe_value = float(probe_work.at[key, label])
                out.at[row_idx, label] = (1.0 - alpha) * base_value + alpha * probe_value
        if missing:
            raise ValueError(f"Probe predictions are missing {len(missing)} baseline rows, e.g. {missing[:5]}")

    out.to_csv(output_csv, index=False)
    print(
        "hybrid replacement saved:",
        output_csv,
        "replaced zero-shot labels:",
        len(replace_labels),
        "alpha:",
        alpha,
    )
    return output_csv


def run_hybrid_zero_shot_replacement_from_models(baseline_csv, model_paths, labels, zero_shot_labels, feature_cfg):
    window_sec = float(feature_cfg.get("window_sec", 5.0))
    pred_cfg = {
        "audio_dir": os.environ.get("HYBRID_PREDICTION_AUDIO_DIR") or os.environ.get("PERCH_PREDICTION_AUDIO_DIR"),
        "filename_col": os.environ.get("HYBRID_FILENAME_COL"),
        "row_id_col": os.environ.get("HYBRID_ROW_ID_COL"),
        "start_col": os.environ.get("HYBRID_START_COL"),
        "end_col": os.environ.get("HYBRID_END_COL"),
        "seconds_col": os.environ.get("HYBRID_SECONDS_COL"),
        "row_id_time_mode": os.environ.get("HYBRID_ROW_ID_TIME_MODE", "end"),
        "window_sec": window_sec,
    }
    baseline_df, pred_rows = build_prediction_rows_from_csv(baseline_csv, DATA_PATH, pred_cfg)
    zeros = np.zeros((len(pred_rows), len(labels)), dtype=np.float32)

    base_name = os.path.splitext(os.path.basename(baseline_csv))[0]
    pred_feature_cfg = dict(feature_cfg)
    pred_feature_cfg["cache_path"] = os.environ.get(
        "PERCH_PREDICTION_FEATURE_CACHE_PATH",
        os.path.join(CACHE_DIR, f"{base_name}_perch_prediction_features.npz"),
    )
    pred_feature_cfg["force_recreate"] = os.environ.get("PERCH_PREDICTION_FEATURE_FORCE_RECREATE", "0") == "1"

    feature_data = extract_or_load_perch_features(pred_rows, labels, zeros, pred_feature_cfg)
    x_pred = build_feature_matrix(feature_data, pred_feature_cfg["feature_mode"])
    probe_pred = predict_with_probe_models(x_pred, model_paths, labels)

    probe_df = pd.DataFrame(probe_pred, columns=labels)
    if "row_id" in baseline_df.columns:
        probe_df.insert(0, "row_id", baseline_df["row_id"].astype(str).values)
    probe_df.insert(0, "start_seconds", pred_rows["start_seconds"].values)
    probe_df.insert(0, "filename", pred_rows["filename"].values)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    probe_csv = os.environ.get(
        "HYBRID_PROBE_PREDICTION_CSV",
        os.path.join(HISTORY_DIR, f"perch_probe_prediction_for_baseline_{timestamp}.csv"),
    )
    output_csv = os.environ.get(
        "HYBRID_OUTPUT_CSV",
        os.path.join(HISTORY_DIR, f"baseline_perch_zero_shot_hybrid_{timestamp}.csv"),
    )
    probe_df.to_csv(probe_csv, index=False)
    print("saved probe prediction CSV:", probe_csv)

    replace_zero_shot_columns(
        baseline_df,
        probe_df,
        labels,
        zero_shot_labels,
        output_csv,
        alpha=float(os.environ.get("HYBRID_ZERO_SHOT_ALPHA", "1.0")),
        positional=True,
        window_sec=window_sec,
    )
    return output_csv


def run_hybrid_zero_shot_replacement_from_csv(baseline_csv, probe_csv, labels, zero_shot_labels):
    baseline_df = pd.read_csv(baseline_csv)
    probe_df = pd.read_csv(probe_csv)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.environ.get(
        "HYBRID_OUTPUT_CSV",
        os.path.join(HISTORY_DIR, f"baseline_perch_zero_shot_hybrid_{timestamp}.csv"),
    )
    return replace_zero_shot_columns(
        baseline_df,
        probe_df,
        labels,
        zero_shot_labels,
        output_csv,
        alpha=float(os.environ.get("HYBRID_ZERO_SHOT_ALPHA", "1.0")),
        positional=os.environ.get("HYBRID_POSITIONAL_REPLACE", "0") == "1",
        window_sec=float(os.environ.get("SOUNDSCAPE_WEAK_WINDOW_SEC", "5.0")),
    )


def parse_path_list(value):
    if not value:
        return []
    return [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]


def run_hybrid_zero_shot_only():
    labels = load_taxonomy_labels(DATA_PATH)
    train_audio_labels = load_train_audio_labels(DATA_PATH)
    zero_shot_labels = [label for label in labels if label not in train_audio_labels]

    baseline_csv = os.environ.get("HYBRID_BASELINE_CSV") or os.environ.get("BASELINE_PREDICTION_CSV")
    if not baseline_csv:
        raise ValueError("HYBRID_ONLY=1 requires HYBRID_BASELINE_CSV or BASELINE_PREDICTION_CSV.")

    probe_csv = os.environ.get("HYBRID_PROBE_CSV") or os.environ.get("PROBE_PREDICTION_CSV")
    if probe_csv:
        return run_hybrid_zero_shot_replacement_from_csv(baseline_csv, probe_csv, labels, zero_shot_labels)

    model_paths = parse_path_list(os.environ.get("PERCH_PROBE_MODEL_PATHS"))
    if not model_paths:
        full_model_path = os.path.join(MODELS_DIR, "perch_feature_probe_full.pt")
        if os.path.exists(full_model_path):
            model_paths = [full_model_path]
        else:
            model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "perch_feature_probe_fold*.pt")))
    if not model_paths:
        raise ValueError(
            "No probe predictions or probe models were found. Set HYBRID_PROBE_CSV, "
            "PROBE_PREDICTION_CSV, or PERCH_PROBE_MODEL_PATHS."
        )

    window_sec = float(os.environ.get("SOUNDSCAPE_WEAK_WINDOW_SEC", "5.0"))
    feature_cfg = {
        "cache_path": None,
        "force_recreate": False,
        "backend": os.environ.get("PERCH_BACKEND", DEFAULT_PERCH_BACKEND),
        "expected_embedding_dim": int(os.environ.get("PERCH_EXPECTED_EMBEDDING_DIM", "1536")),
        "feature_mode": os.environ.get("PERCH_FEATURE_MODE", "emb"),
        "save_logits": os.environ.get("PERCH_FEATURE_SAVE_LOGITS", "0") == "1",
        "batch_size": int(os.environ.get("PERCH_FEATURE_BATCH_SIZE", "16")),
        "sample_rate": int(os.environ.get("PERCH_SAMPLE_RATE", "32000")),
        "window_sec": window_sec,
    }
    return run_hybrid_zero_shot_replacement_from_models(
        baseline_csv,
        model_paths,
        labels,
        zero_shot_labels,
        feature_cfg,
    )


def run_perch_feature_zeroshot_experiment():
    seed = int(os.environ.get("BIRDCLEF_SEED", "42"))
    set_seed(seed)

    labels = load_taxonomy_labels(DATA_PATH)
    train_audio_labels = load_train_audio_labels(DATA_PATH)
    zero_shot_labels = [label for label in labels if label not in train_audio_labels]

    window_sec = float(os.environ.get("SOUNDSCAPE_WEAK_WINDOW_SEC", "5.0"))
    soundscape_cfg = {
        "audio_dir": os.environ.get("SOUNDSCAPE_WEAK_AUDIO_DIR", os.path.join(DATA_PATH, "train_soundscapes")),
        "label_csv": os.environ.get("SOUNDSCAPE_WEAK_LABEL_CSV"),
        "filename_col": os.environ.get("SOUNDSCAPE_WEAK_FILENAME_COL"),
        "row_id_col": os.environ.get("SOUNDSCAPE_WEAK_ROW_ID_COL"),
        "label_col": os.environ.get("SOUNDSCAPE_WEAK_LABEL_COL"),
        "start_col": os.environ.get("SOUNDSCAPE_WEAK_START_COL"),
        "end_col": os.environ.get("SOUNDSCAPE_WEAK_END_COL"),
        "seconds_col": os.environ.get("SOUNDSCAPE_WEAK_SECONDS_COL"),
        "window_sec": window_sec,
        "stride_sec": float(os.environ.get("SOUNDSCAPE_WEAK_STRIDE_SEC", str(window_sec))),
        "file_level_policy": os.environ.get("SOUNDSCAPE_WEAK_FILE_LEVEL_POLICY", "expand"),
        "max_rows": int(os.environ.get("PERCH_FEATURE_MAX_ROWS", "0")),
        "min_per_label": int(os.environ.get("PERCH_FEATURE_MIN_PER_LABEL", "1")),
        "seed": seed,
    }

    print("DATA_PATH:", DATA_PATH)
    print("SOUNDSCAPE_WEAK_LABEL_CSV:", soundscape_cfg["label_csv"] or "auto")
    print("SOUNDSCAPE_WEAK_AUDIO_DIR:", soundscape_cfg["audio_dir"])
    print("PERCH_FEATURE_CACHE_DIR:", CACHE_DIR)
    print("PERCH_FEATURE_MODE:", os.environ.get("PERCH_FEATURE_MODE", "emb"))
    print("zero-shot labels:", len(zero_shot_labels))

    rows = build_soundscape_rows(DATA_PATH, labels, soundscape_cfg)
    y = make_targets(rows, labels)
    positives = y.sum(axis=0)
    covered_labels = [label for label, count in zip(labels, positives) if count > 0]
    covered_zero_shot = [label for label in zero_shot_labels if positives[labels.index(label)] > 0]
    print("covered labels:", len(covered_labels), "covered zero-shot:", len(covered_zero_shot), "/", len(zero_shot_labels))
    if covered_zero_shot:
        print("covered zero-shot labels:", covered_zero_shot[:28])
    missing_zero = [label for label in zero_shot_labels if label not in covered_zero_shot]
    if missing_zero:
        print("missing zero-shot labels:", missing_zero[:28])

    feature_cfg = {
        "cache_path": os.environ.get("PERCH_FEATURE_CACHE_PATH") or None,
        "force_recreate": os.environ.get("PERCH_FEATURE_FORCE_RECREATE", "0") == "1",
        "backend": os.environ.get("PERCH_BACKEND", DEFAULT_PERCH_BACKEND),
        "expected_embedding_dim": int(os.environ.get("PERCH_EXPECTED_EMBEDDING_DIM", "1536")),
        "feature_mode": os.environ.get("PERCH_FEATURE_MODE", "emb"),
        "save_logits": os.environ.get("PERCH_FEATURE_SAVE_LOGITS", "0") == "1",
        "batch_size": int(os.environ.get("PERCH_FEATURE_BATCH_SIZE", "16")),
        "sample_rate": int(os.environ.get("PERCH_SAMPLE_RATE", "32000")),
        "window_sec": window_sec,
    }
    feature_data = extract_or_load_perch_features(rows, labels, y, feature_cfg)
    x = build_feature_matrix(feature_data, feature_cfg["feature_mode"])
    y = feature_data["y"]
    labels = feature_data["labels"]
    filenames = feature_data["filenames"]
    starts = feature_data["start_seconds"]

    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = {
        "device": os.environ.get("PERCH_PROBE_DEVICE", device_default),
        "epochs": int(os.environ.get("PERCH_PROBE_EPOCHS", "80")),
        "train_batch_size": int(os.environ.get("PERCH_PROBE_BATCH_SIZE", "128")),
        "hidden_dims": os.environ.get("PERCH_PROBE_HIDDEN_DIMS", "512,256"),
        "dropout": float(os.environ.get("PERCH_PROBE_DROPOUT", "0.25")),
        "lr": float(os.environ.get("PERCH_PROBE_LR", "1e-3")),
        "weight_decay": float(os.environ.get("PERCH_PROBE_WEIGHT_DECAY", "1e-4")),
        "max_pos_weight": float(os.environ.get("PERCH_PROBE_MAX_POS_WEIGHT", "20")),
        "zero_shot_loss_weight": float(os.environ.get("ZERO_SHOT_LOSS_WEIGHT", "2.0")),
        "patience": int(os.environ.get("PERCH_PROBE_PATIENCE", "20")),
        "log_every": int(os.environ.get("PERCH_PROBE_LOG_EVERY", "5")),
        "feature_mode": feature_cfg["feature_mode"],
    }

    folds = int(os.environ.get("PERCH_PROBE_FOLDS", "5"))
    run_all_folds = os.environ.get("PERCH_PROBE_RUN_ALL_FOLDS", "1") == "1"
    selected_fold = int(os.environ.get("PERCH_PROBE_FOLD", "0"))
    splits = make_splits(
        y,
        groups=filenames,
        folds=folds,
        seed=seed,
        run_all_folds=run_all_folds,
        selected_fold=selected_fold,
    )

    print("feature matrix:", x.shape, "targets:", y.shape)
    print("training device:", train_cfg["device"], "folds:", len(splits), "run_all_folds:", run_all_folds)
    print("zero-shot loss weight:", train_cfg["zero_shot_loss_weight"])

    skip_oof = os.environ.get("PERCH_PROBE_SKIP_OOF", "0") == "1"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = None
    metrics_path = os.path.join(HISTORY_DIR, f"perch_feature_zeroshot_metrics_{timestamp}.csv")
    final_metrics = {}
    oof_pred = np.full_like(y, fill_value=np.nan, dtype=np.float32)
    metrics_rows = []
    model_paths = []
    if skip_oof:
        print("PERCH_PROBE_SKIP_OOF=1: skipping OOF and training only the full Kaggle probe artifact.")
        pd.DataFrame([{"fold": "oof_skipped"}]).to_csv(metrics_path, index=False)
    else:
        for fold, (train_idx, val_idx) in enumerate(splits):
            pred, pred_idx, metrics, model_path = train_probe_fold(
                x,
                y,
                labels,
                zero_shot_labels,
                np.asarray(train_idx),
                np.asarray(val_idx),
                fold,
                train_cfg,
            )
            oof_pred[pred_idx] = pred
            metrics_rows.append({"fold": fold, **metrics})
            model_paths.append(model_path)
            print("fold metrics:", metrics_rows[-1])

        valid_rows = ~np.isnan(oof_pred).any(axis=1)
        final_metrics = evaluate_predictions(y[valid_rows], oof_pred[valid_rows], labels, zero_shot_labels)
        print("OOF metrics:", final_metrics)

        pred_path = os.path.join(HISTORY_DIR, f"perch_feature_zeroshot_oof_{timestamp}.csv")
        save_prediction_csv(pred_path, filenames[valid_rows], starts[valid_rows], labels, oof_pred[valid_rows])
        pd.DataFrame(metrics_rows + [{"fold": "oof", **final_metrics}]).to_csv(metrics_path, index=False)
    print("saved metrics:", metrics_path)

    final_model_paths = list(model_paths)
    full_model_path = None
    artifact_paths = None
    if os.environ.get("PERCH_PROBE_TRAIN_FULL", "1") == "1":
        if os.environ.get("PERCH_PROBE_FULL_EPOCHS"):
            full_epochs = int(os.environ["PERCH_PROBE_FULL_EPOCHS"])
        else:
            best_epochs = [
                int(row["best_epoch"])
                for row in metrics_rows
                if "best_epoch" in row and int(row["best_epoch"]) > 0
            ]
            full_epochs = int(np.median(best_epochs)) if best_epochs else int(train_cfg["epochs"])
        full_epochs = max(1, full_epochs)
        full_model_path = train_probe_full(x, y, labels, zero_shot_labels, train_cfg, full_epochs)
        final_model_paths = [full_model_path]
        artifact_paths = export_probe_artifacts(
            full_model_path=full_model_path,
            metrics_path=metrics_path,
            oof_prediction_path=pred_path,
            labels=labels,
            zero_shot_labels=zero_shot_labels,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            final_metrics=final_metrics,
            full_epochs=full_epochs,
            rows_count=len(y),
            input_dim=x.shape[1],
        )

    hybrid_csv = None
    baseline_csv = os.environ.get("HYBRID_BASELINE_CSV") or os.environ.get("BASELINE_PREDICTION_CSV")
    probe_csv = os.environ.get("HYBRID_PROBE_CSV") or os.environ.get("PROBE_PREDICTION_CSV")
    if baseline_csv:
        print("building hybrid baseline + Perch zero-shot output...")
        if probe_csv:
            hybrid_csv = run_hybrid_zero_shot_replacement_from_csv(baseline_csv, probe_csv, labels, zero_shot_labels)
        else:
            hybrid_csv = run_hybrid_zero_shot_replacement_from_models(
                baseline_csv,
                final_model_paths,
                labels,
                zero_shot_labels,
                feature_cfg,
            )

    return {
        "metrics": final_metrics,
        "fold_metrics": metrics_rows,
        "prediction_csv": pred_path,
        "metrics_csv": metrics_path,
        "model_paths": model_paths,
        "full_model_path": full_model_path,
        "artifact_paths": artifact_paths,
        "hybrid_csv": hybrid_csv,
    }


if __name__ == "__main__":
    if os.environ.get("HYBRID_ONLY", "0") == "1":
        run_hybrid_zero_shot_only()
    else:
        run_perch_feature_zeroshot_experiment()
