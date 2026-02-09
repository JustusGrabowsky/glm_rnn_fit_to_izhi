# glm_rnn_fit_to_izhi

Small collection of scripts to fit Poisson GLMs and train RNNs to reproduce
Izhikevich neuron responses. Implements GLM basis fitting, simulation, and
several RNN model variants; produces comparison plots under `result_plots/`.

Quick start
----------
- Create a Python 3 environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Run the integrated pipeline (all neuron types):

```bash
python run_all.py
```

- Run a single neuron type (example: `1` = tonic_spiking):

```bash
python run_all.py 1
```

- Fit only the GLM or inspect helpers:

```bash
python fit_glm.py   # see file header for arguments and defaults
```

Output
------
Results and figures are written to `result_plots/` (ignored by git).

Notes
-----
- The code is research-oriented and expects reasonably current versions of
  NumPy/SciPy and PyTorch (see `requirements.txt`).
- See individual scripts for more options and parameters.
