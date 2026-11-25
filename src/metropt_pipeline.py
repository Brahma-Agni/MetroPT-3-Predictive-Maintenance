"""Download, prepare, model, and report on the MetroPT-3 compressor data."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
FAILURES = [
    ("2020-04-18 00:00", "2020-04-19 00:00"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]
DIGITAL_COLUMNS = {
    "comp", "dv_eletric", "towers", "mpg", "lps", "pressure_switch", "oil_level",
    "caudal_impulses", "unnamed: 0",
}


def download_dataset(data_dir: Path) -> Path:
    """Download and unpack the UCI archive only when the CSV is missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "MetroPT3(AirCompressor).csv"
    if csv_path.exists():
        return csv_path
    archive = data_dir / "metropt3.zip"
    print(f"Downloading MetroPT-3 from UCI to {archive} ...")
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError("The UCI archive did not contain a CSV file.")
        member = next((name for name in candidates if "aircompressor" in name.lower()), candidates[0])
        with bundle.open(member) as source, csv_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    archive.unlink(missing_ok=True)
    return csv_path


def load_data(path: Path) -> pd.DataFrame:
    """Read and sort source observations, tolerating a BOM in the timestamp header."""
    data = pd.read_csv(path, low_memory=False)
    data.columns = [str(column).replace("\ufeff", "").strip() for column in data.columns]
    timestamp_col = next((c for c in data.columns if c.lower() in {"timestamp", "date"}), None)
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found. Columns: {data.columns.tolist()}")
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    data = data.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def make_features(data: pd.DataFrame, frequency: str, lead_hours: int) -> pd.DataFrame:
    """Create causal rolling statistics and weak labels from reported failure windows."""
    analog = [c for c in data.select_dtypes(include=np.number).columns
              if c.strip().casefold() not in DIGITAL_COLUMNS]
    if not analog:
        raise ValueError("No analog sensor columns were found in the input data.")
    sensors = data[analog].resample(frequency).mean()
    features = {}
    for column in analog:
        series = sensors[column]
        for window in (3, 12):
            rolling = series.rolling(window=window, min_periods=max(2, window // 2))
            features[f"{column}_mean_{window}"] = rolling.mean()
            features[f"{column}_std_{window}"] = rolling.std()
        features[f"{column}_delta_3"] = series - series.shift(3)
    result = pd.DataFrame(features, index=sensors.index)
    labels = pd.Series(0, index=sensors.index, dtype="int8")
    unavailable = pd.Series(False, index=sensors.index)
    lead = pd.Timedelta(hours=lead_hours)
    for start_text, end_text in FAILURES:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        labels.loc[(labels.index >= start - lead) & (labels.index < start)] = 1
        unavailable.loc[(unavailable.index >= start) & (unavailable.index < end)] = True
    result["target"] = labels
    result["failure_period"] = unavailable
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=result.columns.difference(["failure_period"]))


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, max_false_positive_rate: float = 0.01) -> float:
    """Choose the lowest validation threshold that respects the alert-rate budget."""
    negatives = probabilities[np.asarray(y_true) == 0]
    if len(negatives) == 0:
        return 0.5
    for threshold in np.unique(probabilities):
        false_positive_rate = np.mean(negatives >= threshold)
        if false_positive_rate <= max_false_positive_rate:
            return float(threshold)
    return float(np.nextafter(np.max(probabilities), np.inf))


def save_plots(data: pd.DataFrame, test: pd.DataFrame,
               predictions: pd.DataFrame, model: RandomForestClassifier,
               output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, column in zip(axes, ["TP2", "Motor_current"]):
        if column not in data:
            column = data.select_dtypes(include=np.number).columns[0]
        series = data[column].resample("5min").mean()
        axis.plot(series.index, series, linewidth=0.7, color="#2563eb", label=column)
        for start_text, end_text in FAILURES:
            axis.axvspan(pd.Timestamp(start_text), pd.Timestamp(end_text), color="#ef4444", alpha=0.18)
        axis.set_ylabel(column)
        axis.legend(loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("2020")
    fig.suptitle("MetroPT-3 sensor trends and reported failure periods")
    fig.tight_layout()
    fig.savefig(output_dir / "sensor_trends.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(confusion_matrix(test["target"], predictions["failure_prediction"], labels=[0, 1]),
                annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pre-failure"],
                yticklabels=["Normal", "Pre-failure"], ax=axis)
    axis.set(title="Chronological holdout confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).nlargest(15).sort_values()
    fig, axis = plt.subplots(figsize=(9, 7))
    importances.plot.barh(ax=axis, color="#0f766e")
    axis.set(title="Top sensor features", xlabel="Random forest importance", ylabel="")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(15, 5))
    axis.plot(test.index, predictions["failure_probability"], color="#2563eb", linewidth=0.9)
    axis.axhline(predictions.attrs["threshold"], color="#f97316", linestyle="--", label="Validation threshold")
    axis.fill_between(test.index, 0, 1, where=predictions["failure_prediction"].astype(bool),
                      color="#ef4444", alpha=0.12, label="Predicted alert")
    for start_text, end_text in FAILURES:
        axis.axvspan(pd.Timestamp(start_text), pd.Timestamp(end_text), color="#64748b", alpha=0.12)
    axis.set(title="Held-out pre-failure risk scores", ylabel="Failure probability", xlabel="Timestamp", ylim=(0, 1))
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_predictions.png", dpi=150)
    plt.close(fig)

    sensor = "TP2" if "TP2" in data else data.select_dtypes(include=np.number).columns[0]
    trend = data[sensor].resample("5min").mean().reindex(test.index)
    anomaly_mask = predictions["anomaly_flag"].astype(bool)
    fig, axis = plt.subplots(figsize=(15, 5))
    axis.plot(trend.index, trend, color="#2563eb", linewidth=0.8, label=sensor)
    axis.scatter(trend.index[anomaly_mask], trend[anomaly_mask], color="#dc2626", s=14,
                 label="Isolation Forest anomaly", zorder=3)
    for start_text, end_text in FAILURES:
        axis.axvspan(pd.Timestamp(start_text), pd.Timestamp(end_text), color="#64748b", alpha=0.12)
    axis.set(title=f"Held-out {sensor} trend with detected anomalies", ylabel=sensor, xlabel="Timestamp")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "detected_anomalies.png", dpi=150)
    plt.close(fig)


def run(data_path: Path, output_dir: Path, frequency: str, lead_hours: int) -> dict:
    raw = load_data(data_path)
    table = make_features(raw, frequency, lead_hours)
    table = table.loc[~table["failure_period"]].drop(columns="failure_period")
    # Place the final split after the June reports so the held-out period contains
    # the July event while all train/validation rows still precede it in time.
    split_train = int(len(table) * 0.50)
    split_valid = int(len(table) * 0.70)
    train, valid, test = table.iloc[:split_train], table.iloc[split_train:split_valid], table.iloc[split_valid:]
    x_train, y_train = train.drop(columns="target"), train["target"]
    x_valid, y_valid = valid.drop(columns="target"), valid["target"]
    x_test, y_test = test.drop(columns="target"), test["target"]
    if y_train.nunique() < 2 or y_valid.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological splits must each contain normal and pre-failure examples; check date coverage.")

    model = RandomForestClassifier(n_estimators=160, min_samples_leaf=3, class_weight="balanced_subsample",
                                   max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)
    threshold = choose_threshold(y_valid, model.predict_proba(x_valid)[:, 1])
    probabilities = model.predict_proba(x_test)[:, 1]
    predicted = (probabilities >= threshold).astype("int8")
    predictions = pd.DataFrame({"target": y_test, "failure_probability": probabilities,
                                "failure_prediction": predicted}, index=test.index)
    predictions.attrs["threshold"] = threshold
    metrics = {
        "dataset_rows": int(len(raw)), "feature_rows": int(len(table)), "feature_count": int(x_train.shape[1]),
        "frequency": frequency, "lead_hours": lead_hours,
        "train_period": [str(train.index.min()), str(train.index.max())],
        "validation_period": [str(valid.index.min()), str(valid.index.max())],
        "test_period": [str(test.index.min()), str(test.index.max())],
        "validation_selected_threshold": threshold,
        "validation_max_false_positive_rate": 0.01,
        "test_precision": float(precision_score(y_test, predicted, zero_division=0)),
        "test_recall": float(recall_score(y_test, predicted, zero_division=0)),
        "test_f1": float(f1_score(y_test, predicted, zero_division=0)),
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }

    normal_train = x_train.loc[y_train == 0]
    anomaly_model = IsolationForest(n_estimators=100, contamination=0.01, max_samples=10000,
                                    n_jobs=-1, random_state=42).fit(normal_train)
    anomaly_scores = -anomaly_model.score_samples(x_test)
    anomaly_cutoff = float(np.quantile(-anomaly_model.score_samples(normal_train), 0.99))
    predictions["anomaly_score"] = anomaly_scores
    predictions["anomaly_flag"] = (anomaly_scores >= anomaly_cutoff).astype("int8")
    metrics["isolation_forest_test_anomaly_rate"] = float(predictions["anomaly_flag"].mean())
    metrics["isolation_forest_cutoff"] = anomaly_cutoff
    metrics["isolation_forest_precision"] = float(precision_score(y_test, predictions["anomaly_flag"], zero_division=0))
    metrics["isolation_forest_recall"] = float(recall_score(y_test, predictions["anomaly_flag"], zero_division=0))
    metrics["isolation_forest_f1"] = float(f1_score(y_test, predictions["anomaly_flag"], zero_division=0))
    metrics["isolation_forest_average_precision"] = float(average_precision_score(y_test, anomaly_scores))
    metrics["isolation_forest_roc_auc"] = float(roc_auc_score(y_test, anomaly_scores))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "holdout_predictions.csv", index_label="timestamp")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_plots(raw, test, predictions, model, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to the downloaded MetroPT-3 CSV (downloads from UCI if omitted).")
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    parser.add_argument("--frequency", default="5min", help="Sensor aggregation frequency, e.g. 5min.")
    parser.add_argument("--lead-hours", type=int, default=6, help="Pre-failure warning horizon from UCI reports.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    data_path = args.data or download_dataset(args.data_dir)
    metrics = run(data_path, args.output, args.frequency, args.lead_hours)
    print(json.dumps({key: value for key, value in metrics.items() if key.startswith("test_")}, indent=2))
    print(f"\nReports written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
