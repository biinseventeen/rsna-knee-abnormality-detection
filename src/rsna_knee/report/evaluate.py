from __future__ import annotations

import pandas as pd

STATUS_TO_LABEL = {
    "positive": 1.0,
    "negative": 0.0,
    "uncertain": float("nan"),
    "not_mentioned": float("nan"),
}


def evaluate_teacher(
    gold_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    merged = gold_df.merge(
        predicted_df,
        on="StudyInstanceUID",
        suffixes=("_gold", "_pred"),
    )

    rows = []
    for label in labels:
        gold = merged[f"{label}_gold"]
        pred = merged[f"{label}_pred"].map(STATUS_TO_LABEL)

        mask = pred.notna() & gold.notna()
        agreement = (
            (gold[mask].astype(float) == pred[mask]).mean()
            if mask.any()
            else float("nan")
        )

        rows.append({
            "label": label,
            "n_evaluated": int(mask.sum()),
            "exact_agreement": float(agreement),
        })

    return pd.DataFrame(rows)
