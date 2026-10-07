# Snapshot slide evidence map

This map links the current 68-page presentation to the unchanged
[original notebook](stock_alpha_research_pipeline%20copy.ipynb). Stable slide IDs
identify content when page positions change. Notebook cells are zero-based;
the [source manifest](source-manifest.json) records artifact identities.

| Slide ID / current page | Topic | Notebook cells | What the source confirms | Remaining limit |
| --- | --- | --- | --- | --- |
| p29 / 39 | Two research studies | 2, 4, 20–27 | Seven-metric snapshot; actual targets are `ret1y` size and sign; 699 source/696 complete rows. | Unused forward configuration does not establish a forward pipeline. Canonical Quant uses its separate protocol. |
| p30 / 40 | Metric relationships | 7 | Pearson heatmap and displayed correlations are preserved. | Descriptive associations; CSV missing and results not rerun. |
| p31 / 41 | Metric redundancy | 8 | Metric dendrogram uses average linkage and `1 - abs(correlation)`. | Groups metrics, not stocks; no causal or forecasting result. |
| p32 / 42 | PCA compression | 10–13 | Saved variance is 56.87% / 85.64% / 95.84%; component coefficients and three-PC threshold are available. | Exploratory scaler/PCA fit all complete rows; variance is not predictive accuracy. |
| p33 / 43 | Cluster selection | 15–17 | k=2–10; highest silhouette 0.3479 at k=3; seed 42, `n_init=10`. | One snapshot and tested settings; stability across samples/seeds is untested. |
| p34 / 44 | Stock profiles | 17, 18 | Three-cluster means and two-axis PCA scatter are saved. | Projection and means do not establish market regimes or profitable groups. |
| Results divider / 45 | Return size/sign | 19–27 | Introduces the distinct supervised tasks. | No independent empirical result on the divider. |
| p35 / 46 | Magnitude regression | 20, 21 | Random 80/20 split; training-only scaler; exact negative R² and RMSE preserved. | Split code is inspectable; missing CSV prevents reproduction or chronology validation. |
| p36 / 47 | Sign classification | 23–25 | Stratified 80/20 split; 137/140 correct; confusion `[[52,2],[1,85]]`; coefficients saved. | Snapshot sign association; fitted artifacts and per-example predictions unavailable. |
| p37 / 48 | Seven metrics vs three PCs | 23, 26, 27 | Training-only supervised PCA; both saved accuracies are 97.86%. | Equal aggregates do not prove equal predictions; PCA confusion is not saved. |
| p50 / 68 | Acceptance gates | 2, 4, 20–27 | Notebook and split/preprocessing code are now inspectable. | Obtain authorized CSV, timing evidence, and recorded environment; reproduce results before stronger claims. |

The import closes the notebook/code-access gap in earlier slide wording. It
does not close the data or empirical-reproduction gap. Seven extracted PNGs
preserve saved figures without executing models. The archive README distinguishes
all-row exploratory PCA from the separate training-only classification PCA.
