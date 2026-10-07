# Companion metric snapshot archive

This folder preserves the supplied seven-metric stock snapshot notebook and its
saved outputs for inspection and slide traceability. It is a standalone companion
study of metric relationships, stock groups, and one-year return size/sign. The
repository's canonical Quant experiment ranks forward 20-session
benchmark-relative returns under a separate dated, purged protocol. This archive
does not change that protocol or supply results for it.

Open the original [stock_alpha_research_pipeline copy.ipynb](stock_alpha_research_pipeline%20copy.ipynb)
to inspect code and saved outputs. Its original bytes are preserved; its SHA-256 is
`28df5d570bd103513a50124a313c46053993ae97605b848b1c6228bf97410c6d`.
The [source manifest](source-manifest.json) records the source archive hash,
notebook identity, figure hashes, and extraction cells. The
[slide evidence map](slide-evidence.md) connects the notebook to stable slide IDs.

The seven inputs are `sharpe`, `sortino`, `volatility`, `maxDD`, `beta`, `alpha`,
and `infoRatio`. Cell 4 reports 699 rows, 20 columns, and a stock-level snapshot.
Cell 5 reports 699 unique symbols and three missing values in each input;
cell 20 retains 696 complete rows across the inputs and `ret1y`.

## Saved evidence

Cell numbers below are zero-based positions in the original notebook. Values
describe its retained outputs, not a fresh run.

| Evidence | Cells | Saved result / extracted figure |
| --- | --- | --- |
| Input audit | 4, 5, 20 | 699 source rows; 696 complete modelling rows. |
| Metric relationships | 7 | Pearson correlation on complete feature rows; [correlation matrix](figures/correlation-matrix.png). |
| Metric redundancy | 8 | Average-linkage clustering of metrics using `1 - abs(correlation)`; [dendrogram](figures/metric-dendrogram.png). |
| Exploratory PCA | 10–13 | Cumulative variance for 1/2/3 PCs: 56.87% / 85.64% / 95.84%; three PCs meet the 90% threshold; [variance plot](figures/pca-variance.png). |
| Stock clustering | 15–18 | KMeans compares k=2–10 on three PCs; k=3 has the highest saved silhouette, 0.3479; [elbow](figures/kmeans-elbow.png), [silhouette](figures/kmeans-silhouette.png), [stock groups](figures/stock-clusters.png). |
| Return-magnitude regression | 20, 21 | LinearRegression predicts `ret1y`; R² = -1.623931948802356; RMSE = 0.8635017143253988. |
| Return-sign classification | 23 | LogisticRegression predicts `ret1y > 0`; accuracy 0.9785714285714285, or 137/140 = 97.86%; confusion matrix `[[52, 2], [1, 85]]`. |
| Coefficients and compression | 24–27 | [Logistic coefficients](figures/logistic-coefficients.png); three-PC LogisticRegression also reports 137/140 correct. |

## What the code establishes

Exploratory scaling and PCA fit all 696 complete feature rows in cell 10.
The component coefficients are saved in cell 12. KMeans uses the first three exploratory PCs,
`random_state=42`, and `n_init=10`; cell 17 also saves cluster means. These
describe structure within this snapshot and do not measure future prediction.

The supervised paths use random 80/20 splits with seed 42, yielding 556 training
and 140 test examples. The regression split is unstratified. Classification
stratifies by the sign target: 430 positive and 266 nonpositive labels overall,
with 86 positive and 54 nonpositive test labels. Its confusion matrix has actual
classes as rows and predicted classes as columns, ordered 0 then 1.

Both supervised scalers fit training inputs only. Cell 26 fits a separate
three-component PCA to the classification training inputs and transforms its
test inputs. It does not reuse the exploratory all-row PCA. Equal accuracy and
rounded classification reports do not prove identical predictions; a separate
PCA confusion matrix and per-example predictions are not saved. The regression's
negative R² remains a failed magnitude result alongside the high sign accuracy.

Cell 2 declares `FORWARD_DAYS=20` and
`TARGET_COL="forward_20d_excess_return"`. Later modelling cells instead use
`ret1y` and `(ret1y > 0)`. Those unused configuration names do not turn this
snapshot analysis into the canonical forward-ranking pipeline.

## Reproduction boundary

The supplied archive contains no `data.csv`, and the historical environment has
no dependency lock or exact package-version record. The notebook has not been
rerun. Schema/code inspection and byte hashes validate the archived artifact,
not its reported empirical results. No separate provider dataset is included.

Reproduction needs authorized input data, documented metric definitions and
availability times, and a recorded environment. Randomly withholding stocks
does not establish that inputs preceded the one-year outcome. These results
therefore support snapshot associations, not validated future forecasts,
profitable strategies, or promotion to portfolio use. Follow the canonical
[data contract](../../docs/data-contract.md) and
[benchmark protocol](../../docs/stage3-benchmark.md) for new forward studies.
