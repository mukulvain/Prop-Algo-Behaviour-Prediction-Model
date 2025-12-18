# TODO – Prop Algo Behaviour Modeling & LOB Reconstruction

## Current Status (baseline)
- Data: 1-second LOB snapshots for 19 Aug 2019 in `LOB/LOB_19082019.csv`; book builder in `reader.py`/`writer.py` using `Book`/`Ticker`.
- Labels in `train.py`: next-step prop participation (`next_participated`), side, relative price delta, log-qty; simple 80/20 chronological split; XGBoost models trained only on Aug 19; no persistence.
- Features: spread_pct, depth ratios (best & top-5), imbalance, mid return, volatility, skewness, kurtosis, row_total_qty, and lag1/lag2 of imbalance/mid_return/spread_pct.

## Gaps to close
- No simulator to replay RoW vs predicted prop flow and rebuild LOB for holdout days.
- No robust time-series CV; no multi-day training/validation split; no model saving/loading.
- Limited state vector (few flow/temporal features); no trade/order intensity or queue metrics.
- No reconstruction metrics to compare simulated LOB vs actual; no experiment tracking.

## Data & Label Prep
- [ ] Generalize loader to multiple days (19th–30th Aug) and symbols; keep per-day, per-symbol partitions.
- [ ] Ensure prop vs non-prop tagging is consistent from raw orders/trades; validate with spot checks.
- [ ] Build next-step targets at fixed horizon (1s) and allow configurable horizon H; handle market-close boundaries.
- [ ] Persist clean parquet/feather datasets with typed columns and index by event time.

## Feature Engineering (state vector)
- Core price/liq: mid, spread_pct, microprice, mid return (1s, 5s), realized vol (multiple windows), price trend slopes.
- Depth/shape: best depth ratio, deep depth ratio, depth slope across top 5, cumulative depth at top-k, queue length at best, queue aging/turnover.
- Order flow: order arrival/cancel/modify counts per side, net order flow imbalance over short horizons, trade intensity and aggressor side ratio, signed trade volume, $OI = (Q_b - Q_a) / (Q_b + Q_a)$ changes.
- Time features: seconds-from-open, session bucket (pre-open/regular/close), day-of-week if multi-day.
- Prop history: recent prop participation flags, side streaks, inter-arrival times, last prop price offset to mid.
- Regularize with multiple lookbacks (e.g., 1, 5, 10 seconds) and ensure no leakage beyond t.

## Modeling (participation / side / price / qty)
- [ ] Split strategy: rolling/blocked CV by time (e.g., train on 19th, validate on 20th pre-open/early session), no shuffling.
- [ ] Baselines: logistic/LR for participation; majority-class for side; naive "no change" for price; median for qty.
- [ ] Models: compare XGBoost with temporal models (LightGBM, CatBoost, simple GRU/TCN) if sequence info helps.
- [ ] Imbalance handling: tune `scale_pos_weight`, focal loss (if supported), threshold search via PR-AUC.
- [ ] Hyperparam search: small grid/random with early stopping on time-based validation set.
- [ ] Metrics: participation (PR-AUC, ROC-AUC, F1 at optimal threshold), side (accuracy/F1), price delta (MAE/RMSE/QLIKE), qty (MAE/RMSE, coverage of prediction intervals).
- [ ] Persist models + config (json) + feature list + threshold to `models/{date}/`.

## Prop vs RoW Split & Simulator
- [ ] Build a simulator that replays a day’s order flow: inject only non-prop orders to get RoW book trajectory.
- [ ] From RoW book state at t, use models to predict prop participation/side/price/qty for t+1; insert synthetic prop orders and advance book.
- [ ] Allow toggling: actual prop, predicted prop, or none—for ablation.
- [ ] Handle pre-open initialization: seed with 20 Aug pre-open LOB snapshot.

## LOB Reconstruction Metrics (actual vs simulated)
- Price level: RMSE/MAE on mid-price path and spread; hit rate on spread regimes (1-tick, 2-tick, >2-tick).
- Depth/liquidity: MAE on best bid/ask depth, top-5 depth vectors; Wasserstein/earth-mover between depth profiles; imbalance error.
- Trading cost proxies: expected cost to execute x shares (slippage) vs actual; queue position error at best.
- Event timing: participation precision/recall by interval; side accuracy conditional on participation.
- Aggregate: PnL-neutral score combining normalized price/depth errors.

## Backtest Loop (multi-day)
- [ ] Train on day D, simulate day D+1; roll forward window for 10 days; log metrics per day.
- [ ] Track drift: monitor feature distributions and calibration drift of participation probabilities.
- [ ] Maintain leaderboard of configs/hyperparams with metrics snapshot.

## Engineering/Infra
- [ ] Add config file for paths, horizons, model params; set random seeds for reproducibility.
- [ ] Add lightweight experiment logging (csv/json; optional MLflow/W&B if desired).
- [ ] Profiling: measure per-step latency of feature computation and model inference to ensure real-time feasibility.

## Immediate Next Actions
- [ ] Expand feature set in `train.py` (flow, intensity, multi-horizon) and add time-based CV split.
- [ ] Implement model save/load and threshold selection persistence.
- [ ] Prototype simulator that replays RoW only and plugs in participation model outputs for 20 Aug; compute first reconstruction metrics (mid/spread/depth MAE).
