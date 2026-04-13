import os
import glob
import pandas as pd
import numpy as np


BASE_RESULTS_DIR = "results"
OUTPUT_SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, "analysis")

# Methods you want to compare in your presentation
PRESENTATION_METHODS = {"SMGM", "PSO", "GA"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_all_result_csvs(base_dir: str = BASE_RESULTS_DIR) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "*", "*", "logs", "per_image_results.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise FileNotFoundError(f"No result CSV files found with pattern: {pattern}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        parts = os.path.normpath(csv_path).split(os.sep)
        try:
            method_from_path = parts[-4]
            attack_type_from_path = parts[-3]
        except IndexError:
            method_from_path = None
            attack_type_from_path = None

        if "method" not in df.columns or df["method"].isna().all():
            df["method"] = method_from_path
        if "attack_type" not in df.columns or df["attack_type"].isna().all():
            df["attack_type"] = attack_type_from_path

        df["csv_path"] = csv_path
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.strip().str.upper()
    if "attack_type" in df.columns:
        df["attack_type"] = df["attack_type"].astype(str).str.strip().str.lower()
    if "image_name" in df.columns:
        df["image_name"] = df["image_name"].astype(str).str.strip()

    numeric_cols = [
        "clean_num_boxes",
        "adv_num_boxes",
        "clean_score_sum",
        "adv_score_sum",
        "score_reduction",
        "reduction_ratio",
        "attack_success",
        "runtime_sec",
        "l2_perturb",
        "linf_perturb",
        "top_k_boxes",
        "grid_size",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"clean_num_boxes", "adv_num_boxes"}.issubset(df.columns):
        df["box_drop"] = df["clean_num_boxes"] - df["adv_num_boxes"]
        df["box_drop_ratio"] = np.where(
            df["clean_num_boxes"] > 0,
            df["box_drop"] / df["clean_num_boxes"],
            np.nan
        )

    if {"score_reduction", "l2_perturb"}.issubset(df.columns):
        df["score_reduction_per_l2"] = np.where(
            df["l2_perturb"] > 1e-12,
            df["score_reduction"] / df["l2_perturb"],
            np.nan
        )

    if {"score_reduction", "linf_perturb"}.issubset(df.columns):
        df["score_reduction_per_linf"] = np.where(
            df["linf_perturb"] > 1e-12,
            df["score_reduction"] / df["linf_perturb"],
            np.nan
        )

    return df


def deduplicate_per_image_method(df: pd.DataFrame) -> pd.DataFrame:
    required = {"image_name", "method", "attack_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for deduplication: {missing}")

    ranked = df.sort_values(
        by=["image_name", "method", "attack_type", "reduction_ratio", "score_reduction", "runtime_sec"],
        ascending=[True, True, True, False, False, True]
    ).copy()

    dedup_df = (
        ranked.groupby(["image_name", "method", "attack_type"], as_index=False)
        .first()
    )

    return dedup_df


def make_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method", "attack_type"], dropna=False)
        .agg(
            num_images=("image_name", "nunique"),
            success_rate=("attack_success", "mean"),
            mean_clean_boxes=("clean_num_boxes", "mean"),
            mean_adv_boxes=("adv_num_boxes", "mean"),
            mean_box_drop=("box_drop", "mean"),
            mean_box_drop_ratio=("box_drop_ratio", "mean"),
            mean_clean_score=("clean_score_sum", "mean"),
            mean_adv_score=("adv_score_sum", "mean"),
            mean_score_reduction=("score_reduction", "mean"),
            mean_reduction_ratio=("reduction_ratio", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_l2=("l2_perturb", "mean"),
            mean_linf=("linf_perturb", "mean"),
            mean_score_reduction_per_l2=("score_reduction_per_l2", "mean"),
            mean_score_reduction_per_linf=("score_reduction_per_linf", "mean"),
        )
        .reset_index()
    )

    summary["success_rate"] = summary["success_rate"] * 100
    summary = summary.sort_values(
        by=["attack_type", "success_rate", "mean_reduction_ratio"],
        ascending=[True, False, False]
    )
    return summary


def make_best_per_image(df: pd.DataFrame, label_prefix: str = "best") -> pd.DataFrame:
    required = {
        "image_name",
        "method",
        "attack_type",
        "reduction_ratio",
        "score_reduction",
        "runtime_sec",
        "attack_success",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for comparison: {missing}")

    ranked = df.sort_values(
        by=["image_name", "attack_type", "reduction_ratio", "score_reduction", "runtime_sec"],
        ascending=[True, True, False, False, True]
    )

    best_df = ranked.groupby(["image_name", "attack_type"], as_index=False).first()

    best_df = best_df.rename(columns={
        "method": f"{label_prefix}_method",
        "reduction_ratio": f"{label_prefix}_reduction_ratio",
        "score_reduction": f"{label_prefix}_score_reduction",
        "runtime_sec": f"{label_prefix}_runtime_sec",
        "attack_success": f"{label_prefix}_attack_success",
    })

    return best_df


def make_best_method_counts(best_df: pd.DataFrame, method_col: str) -> pd.DataFrame:
    return (
        best_df.groupby(["attack_type", method_col])
        .size()
        .reset_index(name="num_best_images")
        .sort_values(by=["attack_type", "num_best_images"], ascending=[True, False])
    )


def print_duplicate_diagnostics(raw_df: pd.DataFrame, dedup_df: pd.DataFrame):
    print("\n================ DUPLICATE DIAGNOSTICS ================\n")

    raw_counts = (
        raw_df.groupby(["method", "attack_type"])
        .agg(
            raw_rows=("image_name", "count"),
            unique_images=("image_name", "nunique"),
        )
        .reset_index()
    )

    dedup_counts = (
        dedup_df.groupby(["method", "attack_type"])
        .agg(
            dedup_rows=("image_name", "count"),
        )
        .reset_index()
    )

    merged = raw_counts.merge(dedup_counts, on=["method", "attack_type"], how="left")
    merged["removed_rows"] = merged["raw_rows"] - merged["dedup_rows"]

    for _, row in merged.iterrows():
        print(
            f"[{row['method']} | {row['attack_type']}] "
            f"raw_rows={int(row['raw_rows'])}, "
            f"unique_images={int(row['unique_images'])}, "
            f"dedup_rows={int(row['dedup_rows'])}, "
            f"removed_rows={int(row['removed_rows'])}"
        )


def print_pretty_summary(summary_df: pd.DataFrame, title: str):
    print(f"\n================ {title} ================\n")
    for _, row in summary_df.iterrows():
        print(
            f"[{row['method']} | {row['attack_type']}] "
            f"images={int(row['num_images'])}, "
            f"success={row['success_rate']:.2f}%, "
            f"mean_reduction_ratio={row['mean_reduction_ratio']:.4f}, "
            f"mean_box_drop={row['mean_box_drop']:.3f}, "
            f"mean_runtime={row['mean_runtime_sec']:.3f}s, "
            f"mean_l2={row['mean_l2']:.4f}, "
            f"mean_linf={row['mean_linf']:.4f}"
        )


def main():
    ensure_dir(OUTPUT_SUMMARY_DIR)

    # 1) load
    raw_df = load_all_result_csvs(BASE_RESULTS_DIR)
    raw_df = clean_dataframe(raw_df)
    raw_df = add_derived_metrics(raw_df)

    raw_path = os.path.join(OUTPUT_SUMMARY_DIR, "all_results_merged.csv")
    raw_df.to_csv(raw_path, index=False)

    # 2) deduplicate
    dedup_df = deduplicate_per_image_method(raw_df)

    dedup_path = os.path.join(OUTPUT_SUMMARY_DIR, "all_results_deduplicated.csv")
    dedup_df.to_csv(dedup_path, index=False)

    # 3) overall summary
    summary_df = make_method_summary(dedup_df)
    summary_path = os.path.join(OUTPUT_SUMMARY_DIR, "method_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # 4) overall best per image
    overall_best_df = make_best_per_image(dedup_df, label_prefix="best")
    overall_best_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_method_per_image.csv")
    overall_best_df.to_csv(overall_best_path, index=False)

    overall_best_counts_df = make_best_method_counts(overall_best_df, "best_method")
    overall_best_counts_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_method_counts.csv")
    overall_best_counts_df.to_csv(overall_best_counts_path, index=False)

    # 5) presentation-only summary: SMGM, PSO, GA
    presentation_df = dedup_df[dedup_df["method"].isin(PRESENTATION_METHODS)].copy()

    presentation_summary_df = make_method_summary(presentation_df)
    presentation_summary_path = os.path.join(OUTPUT_SUMMARY_DIR, "presentation_summary.csv")
    presentation_summary_df.to_csv(presentation_summary_path, index=False)

    # 6) presentation-only best per image
    presentation_best_df = make_best_per_image(presentation_df, label_prefix="presentation_best")
    presentation_best_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_presentation_method_per_image.csv")
    presentation_best_df.to_csv(presentation_best_path, index=False)

    presentation_best_counts_df = make_best_method_counts(presentation_best_df, "presentation_best_method")
    presentation_best_counts_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_presentation_method_counts.csv")
    presentation_best_counts_df.to_csv(presentation_best_counts_path, index=False)

    # print
    print_duplicate_diagnostics(raw_df, dedup_df)
    print_pretty_summary(summary_df, "METHOD SUMMARY (ALL METHODS)")
    print_pretty_summary(presentation_summary_df, "METHOD SUMMARY (SMGM, PSO, GA)")

    print("\n================ FILES SAVED ================\n")
    print(f"Merged raw results:                 {raw_path}")
    print(f"Merged dedup results:               {dedup_path}")
    print(f"Method summary:                     {summary_path}")
    print(f"Best per image:                     {overall_best_path}")
    print(f"Best method counts:                 {overall_best_counts_path}")
    print(f"Presentation summary:               {presentation_summary_path}")
    print(f"Best presentation method per image: {presentation_best_path}")
    print(f"Best presentation method counts:    {presentation_best_counts_path}")


if __name__ == "__main__":
    main()