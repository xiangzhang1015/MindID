"""
MindID: Person Identification from Brain Waves
through Attention-based Recurrent Neural Network

ACM UbiComp / IMWUT 2018  --  arXiv:1711.06149

Entry point. Runs the full pipeline: data loading, Delta-band
decomposition, attention-based RNN training, and XGBoost evaluation.
"""

import runpy
import os

# Run the model script from the src/ directory so relative data
# paths (EID-M.mat / EID-S.mat) resolve to the project root.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path("src/model.py", run_name="__main__")
