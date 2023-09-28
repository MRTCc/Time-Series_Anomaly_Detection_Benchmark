# Compute Metrics
1. `compute_metrics.py`: 
    - needs the configuration file `config_file.json`
    - given the .npy file with the model estimations,
    - compute: precision, recall, f1-score (and support) and confusion matrices for the requested anomaly rates,
    - then save score in a .npz file.
2. `decode_evaluation.py`: 
    - give the .npz file containing the scores (computed running `compute_metrics.py`),
    - it decode such scores and write them in a human-readable .txt or .csv files (one for each different metric).
    - Note that for the .txt extension, confusion matrices are not decoded, while for the .csv extension they are.
    - The output file names are built in the following way: `metric-name + '_' + outfile-name + .extension`, where the metric names are given by the .npz input file and the others are command line arguments
3. `plot_evaluation.py`: data visualization script
   - needs the configuration file `plot_config.json`
   - plot time-series data channel-wise
   - plot data-estimation-label
   - plot label-estimation-prediction (according to fixed anomaly rate)
   - plot f1_score with respect to the anomaly rate
   - plot f1_pa_score with respect to the anomaly rate
   - plot precision with respect to anomaly rate
   - plot recall with respect to anomaly rate
   - plot confusion matrix (according to a fixed anomaly rate)
   - all plots are saved in .png file that will have the name `out-file + plot-title + .png`; where `out-file` is a configuration parameter.
  
