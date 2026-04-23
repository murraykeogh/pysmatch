# matching.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Union

def perform_match(data: pd.DataFrame, yvar: str, threshold: float = 0.001,
                  nmatches: int = 1, method: str = 'min', replacement: bool = False,
                  sort_by: Union[str, None] = None, round_scores: bool = False, round_value: int = 1) -> pd.DataFrame:

    if 'scores' not in data.columns:
        logging.error("No 'scores' column found. Please run predict_scores() first.")
        raise ValueError("Scores column not found in data.")
    
    if sort_by is not None and sort_by not in data.columns:
        raise ValueError(f"Column '{sort_by}' not found in data.")

    working_data = data.copy()
    
    if round_scores:
        if round_value is None:
            raise ValueError("round_value cannot be None when round_scores is True.")
        if round_value not in range(1, 5):
            raise ValueError("round_value must be between 1 and 4.")
        factor = 10 ** round_value
        working_data['scores'] = (working_data['scores'] * factor).round().astype(int) / factor

    test_df = working_data[working_data[yvar] == 1].copy().reset_index()
    ctrl_df = working_data[working_data[yvar] == 0].copy().reset_index()

    if sort_by is not None and sort_by in test_df.columns and sort_by in ctrl_df.columns:
        test_scores = (test_df[['index', 'scores', sort_by]]
                       .sort_values(['scores', sort_by], ascending=[True, True])
                       .reset_index(drop=True))
        ctrl_scores = (ctrl_df[['index', 'scores', sort_by]]
                       .sort_values(['scores', sort_by], ascending=[True, True])
                       .reset_index(drop=True))
        ctrl_sort_values = ctrl_scores[sort_by].values  # used for explicit tiebreaking
    else:
        if sort_by is not None:
            logging.warning(f"Column '{sort_by}' not found in test or control data. Sorting by scores only.")
        test_scores = (test_df[['index', 'scores']]
                       .sort_values('scores')
                       .reset_index(drop=True))
        ctrl_scores = (ctrl_df[['index', 'scores']]
                       .sort_values('scores')
                       .reset_index(drop=True))
        ctrl_sort_values = None

    test_indices = test_scores['index'].values
    test_scores_values = test_scores['scores'].values.reshape(-1, 1)
    ctrl_indices = ctrl_scores['index'].values
    ctrl_scores_values = ctrl_scores['scores'].values.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=nmatches, radius=threshold, algorithm='ball_tree')
    nbrs.fit(ctrl_scores_values)
    distances, indices = nbrs.radius_neighbors(test_scores_values)

    matched_records = []
    current_match_id = 0
    used_ctrl_indices = set() if not replacement else None

    for i, (dists, neighbors) in enumerate(zip(distances, indices)):
        if len(neighbors) == 0:
            continue

        if method == 'min':
            # Explicitly tiebreak by sort_by values when distances are equal
            if ctrl_sort_values is not None:
                neighbor_sort_vals = np.array([ctrl_sort_values[n] for n in neighbors])
                sorted_order = np.lexsort((neighbor_sort_vals, dists))
            else:
                sorted_order = np.lexsort((neighbors, dists))

            selected = []
            for idx in sorted_order:
                ctrl_idx = ctrl_indices[neighbors[idx]]
                if not replacement and ctrl_idx in used_ctrl_indices:
                    continue
                selected.append(ctrl_idx)
                if not replacement:
                    used_ctrl_indices.add(ctrl_idx)
                if len(selected) == nmatches:
                    break

            if selected:
                for ctrl_idx in selected:
                    matched_records.append({
                        'test_index': test_indices[i],
                        'control_index': ctrl_idx,
                        'match_id': current_match_id
                    })
                current_match_id += 1

        elif method == 'random':
            possible = list(neighbors)
            if not replacement:
                possible = [n for n in possible if ctrl_indices[n] not in used_ctrl_indices]
            if len(possible) == 0:
                continue
            selected = np.random.choice(possible, size=min(nmatches, len(possible)), replace=False)
            for n in selected:
                ctrl_idx = ctrl_indices[n]
                matched_records.append({
                    'test_index': test_indices[i],
                    'control_index': ctrl_idx,
                    'match_id': current_match_id
                })
                if not replacement:
                    used_ctrl_indices.add(ctrl_idx)
            current_match_id += 1
        else:
            raise ValueError("Invalid method parameter, use ('random', 'min')")

    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        matched_test = data.loc[matched_df['test_index']].copy()
        matched_ctrl = data.loc[matched_df['control_index']].copy()
        matched_test['match_id'] = matched_df['match_id'].values
        matched_ctrl['match_id'] = matched_df['match_id'].values
        matched_test['record_id'] = matched_test.index
        matched_ctrl['record_id'] = matched_ctrl.index
        output_df = pd.concat([matched_test, matched_ctrl], ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=list(data.columns) + ['match_id', 'record_id'])

    return output_df


def tune_threshold(data: pd.DataFrame, yvar: str, method: str = 'min',
                   nmatches: int = 1, rng: Union[np.ndarray, None] = None) -> tuple:
    """
    Evaluates matching retention across a range of threshold values.

    Performs matching using `perform_match` for each threshold in the specified
    range (`rng`) and calculates the proportion of the original minority group
    that is retained in the matched dataset. Useful for choosing a threshold.

    Args:
        data (pd.DataFrame): The input DataFrame containing scores and `yvar`.
        yvar (str): The name of the binary treatment/control indicator column.
        method (str, optional): The matching method ('min' or 'random') to use for
                                each evaluation. Defaults to 'min'.
        nmatches (int, optional): The number of matches to seek for each test unit.
                                  Defaults to 1.
        rng (Optional[np.ndarray], optional): A NumPy array of threshold values to test.
                                              If None, defaults to `np.arange(0, 0.001, 0.0001)`.
                                              Defaults to None.

    Returns:
        Tuple[np.ndarray, list]: A tuple containing:
            - thresholds (np.ndarray): The array of threshold values tested.
            - retained (list): A list of proportions (float) of the minority group
                               retained for each corresponding threshold.
    """
    if rng is None:
        rng = np.arange(0, 0.001, 0.0001)
    thresholds = []
    retained = []
    for threshold in rng:
        matched_data = perform_match(data, yvar, threshold=threshold,
                                     nmatches=nmatches, method=method, replacement=False)
        prop = prop_retained(data, matched_data, yvar)
        thresholds.append(threshold)
        retained.append(prop)
    return thresholds, retained


def prop_retained(original_data: pd.DataFrame, matched_data: pd.DataFrame, yvar: str) -> float:
    """
    Calculates the proportion of the minority group retained after matching.

    Compares the number of unique minority group members in the matched dataset
    to the number in the original dataset.

    Args:
        original_data (pd.DataFrame): The dataset before matching.
        matched_data (pd.DataFrame): The dataset after matching. Should contain 'record_id'
                                     or rely on index if 'record_id' is missing.
        yvar (str): The name of the binary treatment/control indicator column.

    Returns:
        float: The proportion (0.0 to 1.0) of the original minority group present
               in the matched dataset. Returns 0.0 if the original minority group
               was empty.
    """
    minority = 1 if (original_data[yvar] == 1).sum() <= (original_data[yvar] == 0).sum() else 0
    denom = len(original_data[original_data[yvar] == minority])
    num = len(matched_data[matched_data[yvar] == minority])
    return num / denom if denom > 0 else 0
