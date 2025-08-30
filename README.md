# PINN-for-PEMFC

This repository contains code for applying Physics-Informed Neural Networks (PINNs) to simulate and analyze Polymer Electrolyte Membrane Fuel Cells (PEMFCs).

## Overview

Physics-Informed Neural Networks (PINNs) are a class of deep learning models that incorporate physical laws described by partial differential equations (PDEs) into the loss function during training. This approach allows for accurate modeling even with limited data.

This project aims to leverage PINNs for the simulation and optimization of PEMFCs, which are widely used in clean energy applications.

## Features

- PINN implementation for PEMFC modeling.
- Jupyter Notebook-based workflows for data pre-processing from simulations.
- Modular code for possible extensions (different PDEs, different neural architectures, different prediction goals etc...).
- Visualization of simulation results.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Recommended libraries: `numpy`, `matplotlib`, `tensorflow` or `pytorch` (depending on your PINN implementation)

### Installation

Clone the repository:
```bash
git clone https://github.com/VinZh22/PINN-for-PEMFC.git
cd PINN-for-PEMFC
```

Install dependencies (example using pip):
```bash
pip install -r requirements.txt
```

### Usage

Open the main notebook and follow the instructions to run the PINN model:

```bash
jupyter notebook
```

## Repository Structure

- `notebooks/` — Jupyter Notebooks for experiments and model training.
- `src/` — Source code modules for PINN and PEMFC simulation.
- `data/` — Input datasets and modeling data.
- `results/` — Output files and visualizations.

## Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

## License

Specify your license here (e.g., MIT, GPL).

## Contact

For questions or collaboration, please contact [VinZh22](https://github.com/VinZh22).
