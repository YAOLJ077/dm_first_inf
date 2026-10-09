# FSIM calibration results

`training_predictions.csv` contains the 30 FSIM training predictions recovered
by rerunning each saved outer-fold parameter configuration on its five training
site-years. Every fold's MAE and RMSE were checked against `fold_parameters.csv`.
These rows are not inferred from the held-out predictions. The latest FSIM-S
results are in `../fsim_s_calibration/`, rather than the older FSIM-S rows in
this directory.

Figure 10 reports the mean of six training-fold metrics and the metrics of the
six pooled held-out predictions. Its error bars are one bootstrap standard
error from 20,000 site-year resamples (seed 20261008), conditional on the fitted
models. A site-year receives the same resampling weight in every training fold;
resamples leaving any training fold empty are excluded from the training
standard errors. No parameter refitting is performed during resampling.
