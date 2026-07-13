import numpy as np
import pandas as pd


def merge_df(
    df_det_fitted,
    manual_corrected_df,
):


    df_det_fitted = df_det_fitted.drop(
        columns=[
            "loss",
            "dark_score",
            "bright_score",
            "mean_outline_distance",
            "yolo_outside_fraction",
            "success",
            "error",
        ])

    manual_corrected_df = manual_corrected_df.drop(
            columns=[
                "loss",
                "dark_score",
                "bright_score",
                "success",
                "error",
            ])

    if "x_manual_refined" in manual_corrected_df.columns:
        manual_corrected_df["x"] = manual_corrected_df["x_manual_refined"].fillna(
            manual_corrected_df["x_old"]
        )
    else:
        manual_corrected_df["x"] = manual_corrected_df["x_old"]

    if "y_manual_refined" in manual_corrected_df.columns:
        manual_corrected_df["y"] = manual_corrected_df["y_manual_refined"].fillna(
            manual_corrected_df["y_old"]
        )
    else:
        manual_corrected_df["y"] = manual_corrected_df["y_old"]


    if "x_refined" in df_det_fitted.columns:
        df_det_fitted["x"] = df_det_fitted["x_refined"].fillna(
            df_det_fitted["x_old"]
        )
    else:
        df_det_fitted["x"] = df_det_fitted["x_old"]

    if "y_refined" in df_det_fitted.columns:
        df_det_fitted["y"] = df_det_fitted["y_refined"].fillna(
            df_det_fitted["y_old"]
        )
    else:
        df_det_fitted["y"] = df_det_fitted["y_old"]

    df_det_fitted = df_det_fitted.drop(
        columns=[
            "x_refined",
            "y_refined",
            "x_old",
            "y_old",
        ],
        errors="ignore",
    )

    manual_corrected_df = manual_corrected_df.drop(
        columns=[
            "x_manual_refined",
            "y_manual_refined",
            "x_old",
            "y_old",
        ],
        errors="ignore",
    )



    final_df = pd.concat(
        [df_det_fitted, manual_corrected_df],
        ignore_index=True,
        sort=False,
    )

    # Give every particle a new unique ID
    final_df["particle_id"] = np.arange(len(final_df))



    return final_df