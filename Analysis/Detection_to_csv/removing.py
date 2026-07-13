

import pandas as pd
import numpy as np


def remove_marked_particles(df_det_fitted, manual_edits_csv, id_col="particle_id"):
    """
    Remove particles from df_det_fitted that were marked for removal
    in the manual edits CSV.

    Parameters
    ----------
    df_det_fitted : pandas.DataFrame
        DataFrame returned by refine_all_particles_to_df.

    manual_edits_csv : str or Path or pandas.DataFrame
        CSV file downloaded from the HTML review tool,
        or already-loaded DataFrame.

    id_col : str
        Name of the particle ID column in df_det_fitted.

    Returns
    -------
    df_clean : pandas.DataFrame
        df_det_fitted with removed particles deleted.

    remove_ids : np.ndarray
        Particle IDs that were removed.
    """

    
    # Load manual edits
    if isinstance(manual_edits_csv, pd.DataFrame):
        manual_edits = manual_edits_csv.copy()
    else:
        manual_edits = pd.read_csv(manual_edits_csv)

    if "remove_particle_id" not in manual_edits.columns:
        raise ValueError("manual edits file must contain column 'remove_particle_id'")

    # Get IDs marked for removal
    remove_ids = (
        manual_edits["remove_particle_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    # Remove those IDs
    df_clean = df_det_fitted[
        ~df_det_fitted[id_col].astype(int).isin(remove_ids)
    ].copy()

    print("Before:", len(df_det_fitted))
    print("Removed:", len(remove_ids))
    print("After :", len(df_clean))

    return df_clean