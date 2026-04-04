import os
import glob
import pandas as pd
import numpy as np


BASE_RESULTS_DIR = "results"
OUTPUT_SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, "analysis")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_all_result_csvs(base_dir: str = BASE_RESULTS_DIR) -> pd.DataFrame:
    """
    Recursively find every per_image_results.csv under:
    results/<method>/<attack_type>/logs/per_image_results.csv
    """
    pattern = os.path.join(base_dir, "*", "*", "logs", "per_image_results.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise FileNotFoundError(
            f"No result CSV files found with pattern: {pattern}"
        )

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # Infer method / attack_type from folder path if needed
        # expected: results/method/attack_type/logs/per_image_results.csv
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

    all_results = pd.concat(frames, ignore_index=True)
    return all_results


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure numeric columns are numeric.
    """
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
    """
    Add a few extra useful metrics for comparison.
    """
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


def make_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by method and attack_type.
    """
    group_cols = ["method", "attack_type"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            num_images=("image_name", "count"),
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


def make_per_image_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare methods on the same image.
    Useful for asking: which method worked best per image?
    """
    required = {"image_name", "method", "attack_type", "reduction_ratio", "runtime_sec", "attack_success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for comparison: {missing}")

    # Pick the best method per image using:
    # 1) higher reduction_ratio
    # 2) if tied, lower runtime
    ranked = df.sort_values(
        by=["image_name", "attack_type", "reduction_ratio", "runtime_sec"],
        ascending=[True, True, False, True]
    )

    best_per_image = ranked.groupby(["image_name", "attack_type"], as_index=False).first()
    best_per_image = best_per_image.rename(
        columns={
            "method": "best_method",
            "reduction_ratio": "best_reduction_ratio",
            "runtime_sec": "best_runtime_sec",
            "attack_success": "best_attack_success"
        }
    )

    return best_per_image


def make_best_method_counts(best_per_image: pd.DataFrame) -> pd.DataFrame:
    """
    Count how often each method was best.
    """
    result = (
        best_per_image.groupby(["attack_type", "best_method"])
        .size()
        .reset_index(name="num_best_images")
        .sort_values(by=["attack_type", "num_best_images"], ascending=[True, False])
    )
    return result


def print_pretty_summary(summary_df: pd.DataFrame):
    """
    Console-friendly printout.
    """
    print("\n================ METHOD SUMMARY ================\n")
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

    df = load_all_result_csvs(BASE_RESULTS_DIR)
    df = clean_dataframe(df)
    df = add_derived_metrics(df)

    # Save merged raw table
    merged_path = os.path.join(OUTPUT_SUMMARY_DIR, "all_results_merged.csv")
    df.to_csv(merged_path, index=False)

    # Method-level summary
    summary_df = make_method_summary(df)
    summary_path = os.path.join(OUTPUT_SUMMARY_DIR, "method_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Per-image best method
    best_per_image_df = make_per_image_comparison(df)
    best_per_image_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_method_per_image.csv")
    best_per_image_df.to_csv(best_per_image_path, index=False)

    # Count wins
    best_counts_df = make_best_method_counts(best_per_image_df)
    best_counts_path = os.path.join(OUTPUT_SUMMARY_DIR, "best_method_counts.csv")
    best_counts_df.to_csv(best_counts_path, index=False)

    print_pretty_summary(summary_df)

    print("\n================ FILES SAVED ================\n")
    print(f"Merged results:      {merged_path}")
    print(f"Method summary:      {summary_path}")
    print(f"Best per image:      {best_per_image_path}")
    print(f"Best method counts:  {best_counts_path}")


if __name__ == "__main__":
    main()