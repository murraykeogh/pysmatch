# matching.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Sequence, Tuple, Union

# A range restriction is expressed as one of:
#   5        -> control value must fall in [test_value - 5, test_value + 5]
#   (-2, 5)  -> offsets added to the test value: [test_value - 2, test_value + 5]
#   '10%'    -> [test_value - 0.1*|test_value|, test_value + 0.1*|test_value|]
RangeSpec = Union[float, int, str, Tuple[float, float], List[float]]


def _parse_range_spec(col: str, spec: RangeSpec) -> Tuple[str, float, float]:
    """
    Normalizes a user supplied range restriction into (kind, lower, upper).

    Args:
        col (str): Column the restriction applies to (used for error messages).
        spec (RangeSpec): Scalar half-width, (lower_offset, upper_offset) pair, or
                          percentage string such as '10%'.

    Returns:
        Tuple[str, float, float]: ``('abs', lower_offset, upper_offset)`` where the
            offsets are added to the test value, or ``('pct', lower_frac, upper_frac)``
            where the fractions are multiplied by the magnitude of the test value.
    """
    if isinstance(spec, str):
        stripped = spec.strip()
        if not stripped.endswith('%'):
            raise ValueError(f"Range for '{col}' given as a string must end with '%' (e.g. '10%'), got '{spec}'.")
        try:
            pct = float(stripped[:-1]) / 100.0
        except ValueError:
            raise ValueError(f"Could not parse percentage range '{spec}' for column '{col}'.")
        if pct < 0:
            raise ValueError(f"Percentage range for '{col}' must be non-negative, got '{spec}'.")
        return 'pct', pct, pct

    if isinstance(spec, (tuple, list, np.ndarray)):
        if len(spec) != 2:
            raise ValueError(f"Range for '{col}' given as a sequence must have exactly 2 elements, got {len(spec)}.")
        lower, upper = float(spec[0]), float(spec[1])
        if lower > upper:
            raise ValueError(f"Range for '{col}' must be (lower_offset, upper_offset) with lower <= upper, got {spec}.")
        return 'abs', lower, upper

    width = float(spec)
    if width < 0:
        raise ValueError(f"Range for '{col}' must be non-negative, got {spec}.")
    return 'abs', -width, width


def _range_bounds(kind: str, lower: float, upper: float, value: float) -> Tuple[float, float]:
    """Returns the (low, high) window of admissible control values for one test value."""
    if kind == 'pct':
        magnitude = abs(value)
        return value - magnitude * lower, value + magnitude * upper
    return value + lower, value + upper


def perform_match(data: pd.DataFrame, yvar: str, threshold: float = 0.001,
                  nmatches: int = 1, method: str = 'min', replacement: bool = False,
                  sort_by: Union[str, None] = None, round_scores: bool = False, round_value: int = 1,
                  exact_match_cols: Optional[Sequence[str]] = None,
                  range_cols: Optional[Dict[str, RangeSpec]] = None) -> pd.DataFrame:
    """
    Matches treated units to control units on propensity score.

    Beyond the propensity score caliper (`threshold`), two optional restrictions can
    shrink the pool of admissible controls for each treated unit:

    * `exact_match_cols`: controls must share identical values with the treated unit on
      every listed column (e.g. sex, state, diagnosis category). Matching is performed
      independently within each stratum, so a control is never paired across strata.
    * `range_cols`: controls must fall inside a window around the treated unit's value
      (e.g. ``{'age': 5}`` for +/- 5 years, ``{'age': (-2, 5)}`` for asymmetric bounds,
      ``{'income': '10%'}`` for +/- 10 percent).

    Args:
        data (pd.DataFrame): Input data containing a 'scores' column and `yvar`.
        yvar (str): Binary treatment indicator column (1 = treated, 0 = control).
        threshold (float, optional): Maximum absolute propensity score difference. Defaults to 0.001.
        nmatches (int, optional): Number of controls to match per treated unit. Defaults to 1.
        method (str, optional): 'min' (closest scores) or 'random'. Defaults to 'min'.
        replacement (bool, optional): Whether a control may be reused. Defaults to False.
        sort_by (Union[str, None], optional): Secondary column used to break score ties. Defaults to None.
        round_scores (bool, optional): Round scores before matching. Defaults to False.
        round_value (int, optional): Decimal places used when `round_scores` is True. Defaults to 1.
        exact_match_cols (Optional[Sequence[str]], optional): Columns requiring an exact match.
                                                              Defaults to None.
        range_cols (Optional[Dict[str, RangeSpec]], optional): Mapping of column name to range
                                                               restriction. Defaults to None.

    Returns:
        pd.DataFrame: Matched treated and control rows with 'match_id' and 'record_id' columns.
                      Empty (with the expected columns) when nothing could be matched.
    """
    if 'scores' not in data.columns:
        logging.error("No 'scores' column found. Please run predict_scores() first.")
        raise ValueError("Scores column not found in data.")

    if sort_by is not None and sort_by not in data.columns:
        raise ValueError(f"Column '{sort_by}' not found in data.")

    if method not in ('min', 'random'):
        raise ValueError("Invalid method parameter, use ('random', 'min')")

    exact_cols = list(exact_match_cols) if exact_match_cols else []
    missing_exact = [col for col in exact_cols if col not in data.columns]
    if missing_exact:
        raise ValueError(f"exact_match_cols not found in data: {missing_exact}")

    parsed_ranges: Dict[str, Tuple[str, float, float]] = {}
    for col, spec in (range_cols or {}).items():
        if col not in data.columns:
            raise ValueError(f"range_cols column '{col}' not found in data.")
        if not is_numeric_dtype(data[col]):
            raise ValueError(f"range_cols column '{col}' must be numeric to define a range.")
        parsed_ranges[col] = _parse_range_spec(col, spec)

    working_data = data.copy()

    if round_scores:
        factor = 10 ** round_value
        working_data['scores'] = (working_data['scores'] * factor).round().astype(int) / factor

    if exact_cols:
        missing_keys = working_data[exact_cols].isna().any(axis=1)
        if missing_keys.any():
            logging.warning(f"Dropping {int(missing_keys.sum())} rows with missing values in "
                            f"exact_match_cols {exact_cols}; exact equality cannot be established for them.")
            working_data = working_data[~missing_keys]

    test_df = working_data[working_data[yvar] == 1].copy().reset_index()
    ctrl_df = working_data[working_data[yvar] == 0].copy().reset_index()

    use_sort_by = sort_by is not None and sort_by in test_df.columns and sort_by in ctrl_df.columns
    if sort_by is not None and not use_sort_by:
        logging.warning(f"Column '{sort_by}' not found in test or control data. Sorting by scores only.")

    # Keep the columns the matching itself needs; dedupe in case of overlap.
    keep_cols = ['index', 'scores']
    for col in ([sort_by] if use_sort_by else []) + list(parsed_ranges) + exact_cols:
        if col not in keep_cols:
            keep_cols.append(col)

    sort_cols = ['scores', sort_by] if use_sort_by else ['scores']
    test_scores = test_df[keep_cols].sort_values(sort_cols).reset_index(drop=True)
    ctrl_scores = ctrl_df[keep_cols].sort_values(sort_cols).reset_index(drop=True)

    matched_records: List[Dict[str, Union[int, float]]] = []
    used_ctrl_indices = set() if not replacement else None
    match_id_counter = [0]

    def match_stratum(test_group: pd.DataFrame, ctrl_group: pd.DataFrame) -> None:
        """Runs nearest-neighbour matching within a single exact-match stratum."""
        if test_group.empty or ctrl_group.empty:
            return

        test_indices = test_group['index'].values
        test_scores_values = test_group['scores'].values.reshape(-1, 1)
        ctrl_indices = ctrl_group['index'].values
        ctrl_scores_values = ctrl_group['scores'].values.reshape(-1, 1)
        ctrl_sort_values = ctrl_group[sort_by].values if use_sort_by else None
        test_range_values = {col: test_group[col].values for col in parsed_ranges}
        ctrl_range_values = {col: ctrl_group[col].values for col in parsed_ranges}

        nbrs = NearestNeighbors(n_neighbors=min(nmatches, len(ctrl_group)),
                                radius=threshold, algorithm='ball_tree')
        nbrs.fit(ctrl_scores_values)
        distances, indices = nbrs.radius_neighbors(test_scores_values)

        for i, (dists, neighbors) in enumerate(zip(distances, indices)):
            if len(neighbors) == 0:
                continue

            if parsed_ranges:
                in_range = np.ones(len(neighbors), dtype=bool)
                for col, (kind, lower, upper) in parsed_ranges.items():
                    test_value = test_range_values[col][i]
                    if pd.isna(test_value):
                        in_range[:] = False
                        break
                    low, high = _range_bounds(kind, lower, upper, test_value)
                    candidate_values = ctrl_range_values[col][neighbors]
                    # NaN control values compare False and are therefore excluded.
                    in_range &= (candidate_values >= low) & (candidate_values <= high)
                if not in_range.any():
                    continue
                neighbors = neighbors[in_range]
                dists = dists[in_range]

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
                            'match_id': match_id_counter[0]
                        })
                    match_id_counter[0] += 1

            else:
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
                        'match_id': match_id_counter[0]
                    })
                    if not replacement:
                        used_ctrl_indices.add(ctrl_idx)
                match_id_counter[0] += 1

    if exact_cols:
        # A length-1 list is deprecated as a groupby key in pandas 2.x; both sides must use
        # the same form so that the stratum keys are comparable.
        group_key = exact_cols[0] if len(exact_cols) == 1 else exact_cols
        ctrl_strata = dict(list(ctrl_scores.groupby(group_key, sort=False)))
        unmatched_strata = 0
        for key, test_group in test_scores.groupby(group_key, sort=False):
            ctrl_group = ctrl_strata.get(key)
            if ctrl_group is None or ctrl_group.empty:
                unmatched_strata += 1
                continue
            match_stratum(test_group, ctrl_group)
        if unmatched_strata:
            logging.warning(f"{unmatched_strata} exact-match stratum/strata had no controls available; "
                            f"treated units in those strata are unmatched.")
    else:
        match_stratum(test_scores, ctrl_scores)

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
                   nmatches: int = 1, rng: Union[np.ndarray, None] = None,
                   exact_match_cols: Optional[Sequence[str]] = None,
                   range_cols: Optional[Dict[str, RangeSpec]] = None) -> tuple:
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
        exact_match_cols (Optional[Sequence[str]], optional): Columns requiring an exact match,
                                                              passed to `perform_match`. Defaults to None.
        range_cols (Optional[Dict[str, RangeSpec]], optional): Range restrictions passed to
                                                               `perform_match`. Defaults to None.

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
                                     nmatches=nmatches, method=method, replacement=False,
                                     exact_match_cols=exact_match_cols, range_cols=range_cols)
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
