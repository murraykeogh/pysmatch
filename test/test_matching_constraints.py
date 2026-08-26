# test_matching_constraints.py
import numpy as np
import pandas as pd
import pytest

from pysmatch.matching import perform_match, _parse_range_spec


@pytest.fixture
def scored_data():
    """Data where every unit has a near-identical score, so constraints drive matching."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        'treated': [1] * (n // 2) + [0] * (n // 2),
        'scores': rng.uniform(0.4, 0.6, size=n),
        'sex': rng.choice(['M', 'F'], size=n),
        'state': rng.choice(['MD', 'VA', 'DC'], size=n),
        'age': rng.randint(20, 80, size=n),
        'income': rng.uniform(20000, 90000, size=n),
    })
    return df


def paired(matched):
    """Returns (case_row, control_row) tuples for each match_id."""
    pairs = []
    for _, grp in matched.groupby('match_id'):
        case = grp[grp['treated'] == 1].iloc[0]
        for _, ctrl in grp[grp['treated'] == 0].iterrows():
            pairs.append((case, ctrl))
    return pairs


def test_exact_match_single_column(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            exact_match_cols=['sex'])
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['sex'] == ctrl['sex']


def test_exact_match_multiple_columns(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            exact_match_cols=['sex', 'state'])
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['sex'] == ctrl['sex']
        assert case['state'] == ctrl['state']


def test_range_scalar_is_symmetric(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            range_cols={'age': 5})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert abs(case['age'] - ctrl['age']) <= 5


def test_range_tuple_is_asymmetric(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            range_cols={'age': (0, 10)})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['age'] <= ctrl['age'] <= case['age'] + 10


def test_range_percentage(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            range_cols={'income': '10%'})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert abs(ctrl['income'] - case['income']) <= 0.1 * abs(case['income']) + 1e-9


def test_exact_and_range_combined(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            exact_match_cols=['sex', 'state'],
                            range_cols={'age': 3, 'income': '20%'})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['sex'] == ctrl['sex']
        assert case['state'] == ctrl['state']
        assert abs(case['age'] - ctrl['age']) <= 3
        assert abs(ctrl['income'] - case['income']) <= 0.2 * abs(case['income']) + 1e-9


def test_constraints_shrink_the_matched_pool(scored_data):
    unconstrained = perform_match(scored_data, 'treated', threshold=0.5)
    constrained = perform_match(scored_data, 'treated', threshold=0.5,
                                exact_match_cols=['sex', 'state'], range_cols={'age': 2})
    assert len(constrained) < len(unconstrained)


def test_no_replacement_holds_across_strata(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5, nmatches=2,
                            replacement=False, exact_match_cols=['state'])
    controls = matched[matched['treated'] == 0]
    assert controls['record_id'].is_unique


def test_replacement_allows_reuse(scored_data):
    matched = perform_match(scored_data, 'treated', threshold=0.5,
                            replacement=True, exact_match_cols=['sex'], range_cols={'age': 1})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['sex'] == ctrl['sex']
        assert abs(case['age'] - ctrl['age']) <= 1


def test_random_method_respects_constraints(scored_data):
    np.random.seed(7)
    matched = perform_match(scored_data, 'treated', threshold=0.5, method='random',
                            exact_match_cols=['sex'], range_cols={'age': 4})
    assert not matched.empty
    for case, ctrl in paired(matched):
        assert case['sex'] == ctrl['sex']
        assert abs(case['age'] - ctrl['age']) <= 4


def test_impossible_exact_stratum_yields_no_matches(scored_data):
    df = scored_data.copy()
    # Treated units are all 'X', controls are all 'Y': no stratum overlap.
    df.loc[df['treated'] == 1, 'sex'] = 'X'
    df.loc[df['treated'] == 0, 'sex'] = 'Y'
    matched = perform_match(df, 'treated', threshold=0.5, exact_match_cols=['sex'])
    assert matched.empty
    assert 'match_id' in matched.columns


def test_rows_with_missing_exact_key_are_dropped(scored_data):
    df = scored_data.copy()
    df.loc[df.index[:5], 'sex'] = np.nan
    matched = perform_match(df, 'treated', threshold=0.5, exact_match_cols=['sex'])
    assert matched['sex'].notna().all()


def test_missing_range_value_excludes_control(scored_data):
    df = scored_data.copy()
    df['age'] = df['age'].astype(float)
    df.loc[df['treated'] == 0, 'age'] = np.nan
    matched = perform_match(df, 'treated', threshold=0.5, range_cols={'age': 5})
    assert matched.empty


def test_no_constraints_matches_previous_behavior(scored_data):
    baseline = perform_match(scored_data, 'treated', threshold=0.01)
    passthrough = perform_match(scored_data, 'treated', threshold=0.01,
                                exact_match_cols=None, range_cols=None)
    pd.testing.assert_frame_equal(baseline, passthrough)


def test_unknown_exact_column_raises(scored_data):
    with pytest.raises(ValueError, match="exact_match_cols not found"):
        perform_match(scored_data, 'treated', exact_match_cols=['nope'])


def test_unknown_range_column_raises(scored_data):
    with pytest.raises(ValueError, match="not found in data"):
        perform_match(scored_data, 'treated', range_cols={'nope': 5})


def test_non_numeric_range_column_raises(scored_data):
    with pytest.raises(ValueError, match="must be numeric"):
        perform_match(scored_data, 'treated', range_cols={'sex': 5})


@pytest.mark.parametrize("spec", [-1, (5, 0), '10', '-5%', (1, 2, 3)])
def test_invalid_range_specs_raise(spec):
    with pytest.raises(ValueError):
        _parse_range_spec('age', spec)


def test_parse_range_spec_forms():
    assert _parse_range_spec('age', 5) == ('abs', -5.0, 5.0)
    assert _parse_range_spec('age', (-2, 5)) == ('abs', -2.0, 5.0)
    assert _parse_range_spec('income', '10%') == ('pct', 0.1, 0.1)
