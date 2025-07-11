Version at time of journal publication: [![DOI code version at paper publication:](https://zenodo.org/badge/DOI/10.5281/zenodo.15863161.svg)](https://doi.org/10.5281/zenodo.15863161)

# FedBatchDesigner

A user-friendly web tool for modeling and optimizing growth-arrested fed-batch bioprocesses.
`FedBatchDesigner` enables rapid exploration of the effect of feed rate and time of switching to the growth-arrested production phase on the TRY metrics (Titer, Rate, Yield) for both constant and exponential feeding strategies.

## Features

- Interactive visualization of productivity&ndash;titer trade-offs
- Support for constant and exponential feeding strategies
- Pre-loaded parameters for common organisms (_E. coli_, _S. cerevisiae_)
- Case study examples including valine and mevalonic acid production in _E. coli_
- Export results and plots for further analysis
- No registration required, no data stored permanently

## Run locally (requires `conda`)

```bash
git clone https://github.com/julibeg/FedBatchDesigner.git
cd FedBatchDesigner
conda env create -f env.yml -n fbd -y
conda activate fbd
shiny run FedBatchDesigner/app.py -b
```

## Usage

1. Input basic process parameters and reactor constraints
2. Provide stage-specific physiological data for your host organism
3. Explore the TRY landscape through interactive visualizations
4. Export results as CSV files and plots as PNG images

## Implementation details

Built with:

- [Shiny for Python](https://github.com/posit-dev/py-shiny)
- [Plotly](https://github.com/plotly/plotly.py)

## Case studies

### L-valine production

- Demonstrates optimization of a two-stage fed-batch process with microaerobic production
- Parameter fitting notebook: `case-studies/valine/fit-parameters.ipynb`

### Ethanol production with enforced ATP wasting

- Demonstrates optimization of a two-stage fed-batch process with nitrogen starvation
- Parameter fitting notebook: `case-studies/ethanol-atp-wasting/fit-parameters.ipynb`

### Mevalonic acid production

- Shows process design for sulfur-limited production
- Parameter fitting notebook: `case-studies/mevalonate/fit-parameters.ipynb`

## Model assumptions

1. Single substrate limitation
2. Zero substrate concentration during feed phases
3. Constant stage-specific host characteristics
4. Linear relationship between product formation and growth rate ($\pi = \pi_0 + \mu \cdot \pi_1$)
5. Negligible product formation during batch phase
6. Growth determined by remaining substrate after maintenance and production
7. Complete growth arrest in production stage
8. Neglected volume changes from base addition/evaporation
9. TRY metrics based on feed phase only

## License

This project is licensed under the MIT License - see the LICENSE file for details.
