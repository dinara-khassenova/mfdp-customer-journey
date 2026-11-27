"""Compare adapted ranking methods on RetailRocket, without ML.

Run: python scripts/retailrocket_priority_benchmark_v2.py
     --events data/raw/events.csv --out results/task3/v2

Vendor-inspired methods are not tests of the original products.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)
GROUPS = [
    '00_no_views', '01_one_view', '02_2_3_one_item', '03_2_3_multi_item',
    '04_4_9_one_item', '05_4_9_multi_item', '06_10plus_one_item',
    '07_10plus_multi_item',
]
COMPETITORS = ['Mixpanel-inspired', 'Contentsquare-inspired', 'Fullstory-inspired']
AVITO = 'Big10 Pulse-inspired (reference only)'
BASELINE = 'Baseline (fixed initial ranking)'
METHODS = COMPETITORS + [AVITO, BASELINE]
METRICS = [
    'hit_at_1_pct', 'impact_capture_at_1_pct', 'impact_capture_at_3_pct',
    'ndcg_at_3', 'recommendation_rate_pct',
]


def make_sessions(path):
    """Keep the session, group and target definitions from experiment v1."""
    events = pd.read_csv(path)
    required = {'timestamp', 'visitorid', 'event', 'itemid'}
    if not required.issubset(events.columns):
        raise ValueError(f'Missing columns: {required - set(events.columns)}')
    raw_n = len(events)
    events = events.drop_duplicates().copy()
    if events.empty or events[list(required)].isna().any().any():
        raise ValueError('Empty data or missing required event fields.')
    events['time'] = pd.to_datetime(events.timestamp, unit='ms', utc=True)
    events = events.sort_values(['visitorid', 'time'], kind='stable')
    start, end = events.time.min(), events.time.max()
    new_session = (
        events.visitorid.ne(events.visitorid.shift())
        | events.time.diff().gt(pd.Timedelta(minutes=30))
    )
    events['session_id'] = new_session.cumsum()
    cart = (
        events.loc[events.event.eq('addtocart'), ['session_id', 'visitorid', 'time']]
        .drop_duplicates('session_id')
        .rename(columns={'time': 'cart_time'})
    )
    views = events.loc[events.event.eq('view'), ['session_id', 'time', 'itemid']]
    views = views.merge(cart[['session_id', 'cart_time']], on='session_id')
    views = views.loc[views.time.lt(views.cart_time)]
    counts = views.groupby('session_id').agg(
        views=('itemid', 'size'), unique_items=('itemid', 'nunique')
    )
    cart = cart.join(counts, on='session_id')
    columns = ['views', 'unique_items']
    cart[columns] = cart[columns].fillna(0).astype(int)

    # Same visitor, any purchased item, strictly after the first cart event.
    purchases = (
        events.loc[events.event.eq('transaction'), ['visitorid', 'time']]
        .rename(columns={'time': 'purchase_time'})
        .sort_values('purchase_time')
    )
    cart = pd.merge_asof(
        cart.sort_values('cart_time'), purchases, by='visitorid',
        left_on='cart_time', right_on='purchase_time', direction='forward',
        tolerance=DAY, allow_exact_matches=False,
    )
    cart = cart.loc[cart.cart_time.le(end - DAY)].copy()
    cart['abandon'] = cart.purchase_time.isna().astype(int)
    n, u = cart.views, cart.unique_items
    conditions = [
        n.eq(0), n.eq(1), n.between(2, 3) & u.eq(1),
        n.between(2, 3) & u.gt(1), n.between(4, 9) & u.eq(1),
        n.between(4, 9) & u.gt(1), n.ge(10) & u.eq(1),
    ]
    cart['group'] = np.select(conditions, GROUPS[:-1], default=GROUPS[-1])
    audit = {
        'raw_events': raw_n,
        'duplicate_events_removed': raw_n - len(events),
        'sessions_30min': events.session_id.nunique(),
        'complete_cart_sessions': len(cart),
        'abandon_rate': cart.abandon.mean(),
        'data_start': start,
        'data_end': end,
    }
    return cart, start, end, audit


def period(data, start, end):
    """Select a half-open time interval [start, end)."""
    return data.loc[data.cart_time.ge(start) & data.cart_time.lt(end)]


def group_stats(data):
    """Compare each group with all other cart sessions in the same period."""
    stats = data.groupby('group').abandon.agg(n='size', failures='sum')
    stats = stats.reindex(GROUPS, fill_value=0)
    stats['risk'] = stats.failures / stats.n.replace(0, np.nan)
    stats['rest_n'] = len(data) - stats.n
    stats['rest_risk'] = (
        (data.abandon.sum() - stats.failures) / stats.rest_n.replace(0, np.nan)
    )
    stats['gap'] = stats.risk - stats.rest_risk
    stats['impact'] = (stats.n * stats.gap.clip(lower=0)).fillna(0)

    # One-sided pooled two-proportion z statistic. Session-level approximation:
    # it does not correct for repeated visitors or multiple comparisons.
    pooled = data.abandon.mean()
    se = np.sqrt(pooled * (1 - pooled) * (
        1 / stats.n.replace(0, np.nan) + 1 / stats.rest_n.replace(0, np.nan)
    ))
    stats['z'] = stats.gap / se.replace(0, np.nan)
    return stats


def historical_scores(history, first_half, second_half, min_group):
    """Calculate all scores before validation data are opened."""
    stats = group_stats(history)
    eligible = stats.index[
        stats.n.ge(min_group) & stats.rest_n.ge(min_group)
    ]
    scores = pd.DataFrame(index=eligible)
    # Signal: signed Phi, with the binary goal changed to cart abandonment.
    q = stats.n / len(history)
    p = history.abandon.mean()
    if 0 < p < 1:
        phi = stats.gap * np.sqrt(q * (1 - q) / (p * (1 - p)))
    else:
        phi = 0.0
    scores[COMPETITORS[0]] = phi

    # Impact Evaluation: estimated missed conversions; apply the documented
    # z > 2.33 confidence criterion as a recommendation filter in this test.
    scores[COMPETITORS[1]] = stats.impact.where(stats.z.gt(2.33))

    # Fullstory: relative conversion impact, not its default lost-conversion
    # sort. 95% one-sided z threshold is our explicit approximation, not a
    # claim about Fullstory's proprietary significance algorithm or threshold.
    relative_drop = stats.gap / (1 - stats.rest_risk).replace(0, np.nan)
    scores[COMPETITORS[2]] = relative_drop.where(stats.z.gt(1.644854))

    # Avito-inspired: exact additive contribution to the CHANGE in overall
    # abandonment rate (includes both mix and within-group rate effects).
    # Reference only: the actual Avito service uses financial metrics.
    old = group_stats(first_half)
    new = group_stats(second_half)
    contribution = (
        new.failures / len(second_half) - old.failures / len(first_half)
    )
    scores[AVITO] = contribution
    return scores, stats


def ranked(scores):
    """Positive scores only; ties resolved by the fixed group identifier."""
    scores = scores.dropna()
    scores = scores.loc[scores.gt(0)]
    return sorted(scores.index, key=lambda group: (-scores[group], group))


def ranking_metrics(recommended, actual):
    """Compare recommendations with the same observed impact for all methods."""
    ideal = ranked(actual)
    result = dict.fromkeys(METRICS, np.nan)
    result['recommendation_rate_pct'] = 100.0 * bool(recommended)
    if not ideal:
        # No positive validation impact: ranking quality is undefined.
        return result, ideal

    first_impact = actual.loc[recommended[0]] if recommended else 0.0
    best_impact = actual.loc[ideal[0]]
    # Any group tied for the maximum is a correct first recommendation.
    result['hit_at_1_pct'] = 100.0 * bool(
        recommended and np.isclose(first_impact, best_impact)
    )
    result['impact_capture_at_1_pct'] = 100 * first_impact / best_impact
    gains = actual.reindex(recommended[:3]).to_numpy()
    best = actual.loc[ideal[:3]].to_numpy()
    weights = 1 / np.log2(np.arange(2, 5))
    result['impact_capture_at_3_pct'] = 100 * gains.sum() / best.sum()
    result['ndcg_at_3'] = (
        np.dot(gains, weights[:len(gains)]) / np.dot(best, weights[:len(best)])
    )
    return result, ideal


def run(cart, start, end, weeks, min_group):
    # The discovery interval ends one day before validation starts so that
    # every 24-hour discovery label is already available when we rank groups.
    final = (end - DAY).floor('D')
    first_test = final - weeks * WEEK
    if first_test - DAY - 2 * WEEK < start:
        raise ValueError('Not enough history. Reduce --weeks.')
    records, details, audits = [], [], []
    fixed_scores = None
    for i in range(weeks):
        test_start = first_test + i * WEEK
        test_end = test_start + WEEK
        cutoff = test_start - DAY
        discovery_start = cutoff - 2 * WEEK
        older = period(cart, discovery_start, cutoff - WEEK)
        newer = period(cart, cutoff - WEEK, cutoff)
        history = pd.concat([older, newer])
        audit = {
            'window': test_start.date().isoformat(),
            'discovery_start': discovery_start.isoformat(),
            'discovery_end_exclusive': cutoff.isoformat(),
            'test_end_exclusive': test_end.isoformat(),
            'history_n': len(history),
        }
        if older.empty or newer.empty:
            audits.append({**audit, 'status': 'skipped: no discovery data'})
            continue
        scores, hist_stats = historical_scores(history, older, newer, min_group)
        if len(scores) < 2:
            audits.append({**audit, 'status': 'skipped: insufficient groups'})
            continue
        # Freeze a simple, unfiltered impact ranking on the first valid history.
        # No future outcomes or group sizes are used, even for eligibility.
        if fixed_scores is None:
            fixed_scores = hist_stats.loc[scores.index, 'impact'].copy()
        scores[BASELINE] = fixed_scores.reindex(scores.index)
        recommendations = {method: ranked(scores[method]) for method in METHODS}

        # Only now use the next week to score the already frozen recommendations.
        test = period(cart, test_start, test_end)
        audit.update(test_n=len(test), eligible_groups=len(scores))
        if test.empty or test.group.nunique() < 2:
            audits.append({**audit, 'status': 'skipped: no comparison in test'})
            continue
        actual_stats = group_stats(test)
        actual = actual_stats.loc[scores.index, 'impact']
        audit.update(
            positive_groups=int(actual.gt(0).sum()),
            zero_test_groups=int(actual_stats.loc[scores.index, 'n'].eq(0).sum()),
            status='ok' if actual.gt(0).any() else 'no positive validation impact',
        )
        audits.append(audit)
        for method, recommendation in recommendations.items():
            result, ideal = ranking_metrics(recommendation, actual)
            records.append({
                'window': audit['window'], 'method': method, **result,
                'recommended_top3': ' | '.join(recommendation[:3]),
                'actual_top3': ' | '.join(ideal[:3]),
                'recommended_count': len(recommendation),
                'best_impact': actual.max(),
            })
            ranks = {group: rank for rank, group in enumerate(recommendation, 1)}
            for group in scores.index:
                details.append({
                    'window': audit['window'], 'method': method, 'group': group,
                    'rank': ranks.get(group), 'score': scores.loc[group, method],
                    'history_n': hist_stats.loc[group, 'n'],
                    'history_risk': hist_stats.loc[group, 'risk'],
                    'history_rest_risk': hist_stats.loc[group, 'rest_risk'],
                    'history_z': hist_stats.loc[group, 'z'],
                    'test_n': actual_stats.loc[group, 'n'],
                    'actual_risk': actual_stats.loc[group, 'risk'],
                    'actual_rest_risk': actual_stats.loc[group, 'rest_risk'],
                    'actual_impact': actual[group],
                })
    if not records:
        raise ValueError('No valid test windows. Check history and group sizes.')
    return pd.DataFrame(records), pd.DataFrame(details), pd.DataFrame(audits)


def summarize(records):
    """Only the three comparable competitor methods enter the benchmark."""
    grouped = records.groupby('method')[METRICS]
    means = grouped.mean().reindex(METHODS)
    counts = grouped.count().reindex(METHODS).add_suffix('_n_windows')
    competitors = means.loc[COMPETITORS]
    n_methods = competitors.count()
    extra = pd.DataFrame([
        competitors.mean().where(n_methods >= 3),
        competitors.max().where(n_methods >= 3),
    ], index=['Benchmark mean (>=3 methods)', 'Benchmark best (>=3 methods)'])
    summary = pd.concat([means.join(counts), extra])
    summary.index.name = 'method'
    return summary, n_methods.rename('n_methods')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/task3/v2'))
    parser.add_argument('--weeks', type=int, default=6)
    parser.add_argument('--min-group', type=int, default=20)
    args = parser.parse_args()
    if args.weeks < 1 or args.min_group < 1:
        raise ValueError('--weeks and --min-group must be positive.')
    args.out.mkdir(parents=True, exist_ok=True)
    cart, start, end, audit = make_sessions(args.events)
    records, details, windows = run(cart, start, end, args.weeks, args.min_group)
    summary, n_methods = summarize(records)
    pd.DataFrame([audit]).to_csv(args.out / 'data_audit.csv', index=False)
    windows.to_csv(args.out / 'window_audit.csv', index=False)
    records.to_csv(args.out / 'metrics_by_window.csv', index=False)
    details.to_csv(args.out / 'group_rankings.csv', index=False)
    summary.to_csv(args.out / 'benchmark_summary.csv', float_format='%.6f')
    n_methods.to_csv(args.out / 'benchmark_n_methods.csv')
    print('\nDATA AUDIT\n', pd.Series(audit).to_string())
    print('\nWINDOW AUDIT\n', windows.to_string(index=False))
    print('\nBENCHMARK\n', summary[METRICS].round(4).to_string())
    print('\nNUMBER OF COMPETITOR METHODS\n', n_methods.to_string())
    print('\nFiles saved to:', args.out.resolve())


if __name__ == '__main__':
    main()
