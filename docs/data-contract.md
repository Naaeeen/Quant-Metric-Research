# Data contract

## Decision time

The MVP assumes signals are formed after the `as_of_date` market close. An
`entry_lag_sessions` value of one therefore starts the label on the next
benchmark session. A horizon of 20 means 20 close-to-close benchmark-session
intervals from that entry point. Both dates are stored in every output row.

## Price input

Required columns:

- `date`: normalized trading date.
- `symbol`: stable security identifier used by this dataset version.
- `adjusted_close`: finite, strictly positive adjusted close.

Rows must be unique by `(date, symbol)`. The benchmark must be present in the
same table. The core library does not download or silently repair market data;
provider-specific ingestion belongs in an adapter with its own provenance.

## Universe membership input

Required columns:

- `universe_id`
- `symbol`
- `effective_from`
- `effective_to` (nullable)
- `source`

Intervals are half-open: `effective_from <= as_of_date < effective_to`. Null
`effective_to` means open-ended. Intervals for the same universe and symbol may
not overlap. A current constituent list without dated intervals is allowed only
as an explicitly labelled demo and cannot support an unbiased historical claim.

## Panel output

One row represents one active security at one decision date. Metadata includes:

- dataset version and universe identifier;
- feature availability, window start/end, observation count, and eligibility;
- benchmark symbol and metric configuration;
- label start/end, label availability, forward return, benchmark return,
  excess return, and cross-sectional target rank.

The stored configuration includes lookback, minimum observations, target
horizon, entry lag, annualization frequency, and annual risk-free rate so a
panel row can be traced to the assumptions that produced it.

The initial feature family mirrors reusable per-security metrics from the
existing application: trailing return, Sharpe, Sortino, volatility, maximum
drawdown, beta, CAPM alpha, information ratio, benchmark correlation, and
historical VaR. Efficient frontier output is excluded because it is a
portfolio-level optimization result, not a stable per-security feature.

## Missingness

Rows are retained when prices, history, or labels are unavailable. Numeric
fields remain null and a reason column records why. This prevents silent
survivor filtering and makes coverage part of Stage 2 analysis.

## Split boundary

All training folds must end before the first test decision date, and every
training row must also have `label_end_date < first_test_as_of_date`. This
purges overlapping forward labels without assuming a fixed sampling frequency.

Stage 2's optional walk-forward report records the train end, maximum retained
label end, test start/end, train selection status, test Rank IC, and test
quantile spread for every feature and fold. It also applies the training
direction to test Rank IC, so a stable negative predictor is not mistaken for
an unstable or useless one.
