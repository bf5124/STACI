\# STACI: Spatio-Temporal Aleatoric Conformal Inference



> Recreates the STACI paper results.



\## Environment

Packages are pinned in `environment.yml`.



\## Data

We include datasets under `data/`:

\- `AOD\_data/`

\- `MSS\_data/`



\## How to Run

The main entry points are:

1\. `main\_AOD.py` — Air pollution (AOD) dataset

2\. `main\_MSS.py` — Synthetic Mean Sea Surface (MSS) dataset



Results are saved to the `results/` folder.



Set desired arguments in `args.py`, and adjust FFN latent model parameters in `model.py`.



\## Code Map

\- `BayesNN.py` — STACI neural network approximation

\- `svgd.py` — SVGD training code

\- `conformal.py` — Conformal fitting and cross-validation



\## Quickstart

```bash

\# After creating/activating the conda env from environment.yml

python main\_AOD.py

python main\_MSS.py

