# Data

MindID uses three EEG datasets. Two are local datasets distributed with this repository; one is a publicly available dataset from PhysioNet.

## Datasets

### EID-M (local)

A local dataset collected by the authors. It contains EEG recordings from 8 subjects across 3 trials, yielding 21,000 samples per subject (64 channels, sampled at 128 Hz).

File: `EID-M.mat`
MATLAB variable: `eeg_close_ubicomp_8sub`

### EID-S (local)

A local dataset with 1 trial, providing 7,000 samples per subject across 8 subjects (14 channels, sampled at 128 Hz). This is the dataset used by default in `src/model.py`.

File: `EID-S.mat`
MATLAB variable: `eeg_close_8sub_1file`

### EEG-S (public)

A subset of the EEG Motor Movement/Imagery Dataset (eegmmidb) from PhysioNet. The subset selection and preprocessing code is included in `src/model.py`.

- Source: [PhysioNet eegmmidb](https://www.physionet.org/pn4/eegmmidb/)
- MATLAB variable after download: `EEG_ID_label6`
- Sampling rate: 160 Hz; 64 channels; 13,500 samples per subject (8 subjects used)

## File Placement

Place all `.mat` files in the project root directory (next to `main.py`):

```
MindID/
  main.py
  EID-M.mat        <-- here
  EID-S.mat        <-- here
  src/
  data/
```

## Switching Datasets

Open `src/model.py` and edit the data-loading block (lines 46-49). To use EID-M, uncomment:

```python
feature = sc.loadmat("EID-M.mat")
all = feature['eeg_close_ubicomp_8sub']
```

and comment out the EID-S lines. Adjust `n_fea` and `full` accordingly (see comments in the file).

To use EEG-S (PhysioNet), download the eegmmidb dataset, convert to `EEG_ID_label6.mat`, and follow the commented block starting at line 56 in `src/model.py`.
