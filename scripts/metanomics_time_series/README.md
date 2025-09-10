# Yeast Metabolomics Time Series Data

This directory contains time series metabolomics and gene expression data from yeast glucose pulse experiments, converted from the original Excel supplementary data to machine-readable CSV format.

## Data Source

The original data comes from supplementary materials of a Systems Biology study on yeast glucose pulse response. The data appears to be from Molecular Systems Biology publication msb4100083 and includes:

- Metabolite concentration measurements (intracellular and extracellular)
- Nucleotide pool dynamics
- Gene expression changes over time

## Experimental Design

**Experiment Type**: Glucose pulse experiment on yeast
**Conditions**: 3 independent experimental pulses (replicates)
**Time Range**: -100 to 361.5 seconds (pre-pulse baseline and post-pulse response)
**Measurements**: Concentrations measured at multiple time points

## Files and Data Structure

### 1. `nucleotides_timeseries.csv`
**Description**: Nucleotide concentration time series
**Rows**: 44 (time points across 3 pulse experiments)
**Columns**: 18 (dataset, time, + 8 nucleotides × 2 statistics)

**Nucleotides measured**:
- ATP, ADP, AMP (adenosine nucleotides)
- CTP, CMP (cytidine nucleotides)
- GDP (guanosine nucleotide)
- UTP, UMP (uridine nucleotides)

**Units**: μmol/gDW (micromoles per gram dry weight)

**Column format**:
- `dataset`: pulse_1, pulse_2, or pulse_3
- `time_s`: Time in seconds (negative = before glucose pulse)
- `{nucleotide}_mean`: Average concentration
- `{nucleotide}_std`: Standard deviation

### 2. `intracellular_metabolites_timeseries.csv`
**Description**: Intracellular metabolite concentration time series
**Rows**: 44 (time points across 3 pulse experiments)
**Columns**: 32 (dataset, time, + 15 metabolites × 2 statistics)

**Metabolites measured** (grouped by pathway):
- **Glycolysis**: G6P, F6P, F1,6P2, F2,6P2, 23PG, PEP
- **Pentose phosphate**: 6PG, G1P, T6P
- **TCA cycle**: Cit/icit, OGL, SUC, FUM, MAL
- **Redox**: NAD/NADH

**Units**: μmol/gDW (micromoles per gram dry weight)

**Column format**:
- `dataset`: pulse_1, pulse_2, or pulse_3
- `time_s`: Time in seconds
- `{metabolite}_mean`: Average concentration
- `{metabolite}_cv_percent`: Coefficient of variation (%)

### 3. `extracellular_metabolites_timeseries.csv`
**Description**: Extracellular metabolite concentration time series
**Rows**: 42 (time points across 3 pulse experiments)
**Columns**: 10 (dataset, time, + 4 metabolites × 2 statistics)

**Metabolites measured**:
- **glucose**: Primary carbon source
- **ethanol**: Fermentation product
- **acetate**: Organic acid byproduct
- **glycerol**: Osmotic regulator/byproduct

**Units**: mM (millimolar)

**Column format**:
- `dataset`: pulse_1, pulse_2, or pulse_3
- `time_s`: Time in seconds
- `{metabolite}_mean`: Average concentration
- `{metabolite}_cv_percent`: Coefficient of variation (%)

### 4. `gene_expression_timeseries.csv`
**Description**: Gene expression changes during glucose pulse
**Rows**: 1157 (genes measured on Affymetrix arrays)
**Columns**: 33 (gene info + time points + derived metrics)

**Time points**: T=0, 30, 60, 120, 210, 300, 330 seconds

**Units**: Affymetrix signal intensity (arbitrary units)

**Key columns**:
- `probeset_id`: Affymetrix probe set identifier
- `gene_name`: Yeast systematic gene name (e.g., YLR333C)
- `T{time}s_avg`: Average expression at time point (seconds)
- `T{time}s_std`: Standard deviation of expression
- `fold_change_30vs0`: Fold change from T=0 to T=30

## Data Quality Notes

- **Missing values**: Represented as NaN, particularly common in early time points
- **Replicates**: 3 independent pulse experiments provide biological replicates
- **Time resolution**: Higher density sampling in first 60 seconds post-pulse
- **Detection limits**: Some metabolites below detection at certain time points

## Usage Examples

### Loading the data in Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load nucleotide data
nucleotides = pd.read_csv('nucleotides_timeseries.csv')

# Filter to first pulse experiment
pulse1 = nucleotides[nucleotides.dataset == 'pulse_1']

# Plot ATP dynamics
plt.figure(figsize=(8, 5))
plt.plot(pulse1.time_s, pulse1.ATP_mean, 'o-', label='ATP')
plt.fill_between(pulse1.time_s,
                 pulse1.ATP_mean - pulse1.ATP_std,
                 pulse1.ATP_mean + pulse1.ATP_std, alpha=0.3)
plt.axvline(0, color='red', linestyle='--', label='Glucose pulse')
plt.xlabel('Time (seconds)')
plt.ylabel('ATP concentration (μmol/gDW)')
plt.legend()
plt.title('ATP Response to Glucose Pulse')
plt.show()
```

### Analyzing metabolite correlations

```python
# Load intracellular metabolites
metabolites = pd.read_csv('intracellular_metabolites_timeseries.csv')

# Get concentration columns only (exclude time/dataset)
conc_cols = [col for col in metabolites.columns if col.endswith('_mean')]
conc_data = metabolites[conc_cols].dropna()

# Compute correlation matrix
corr_matrix = conc_data.corr()
print("Metabolite correlations:")
print(corr_matrix.round(2))
```

### Time series analysis for LLRQ modeling

```python
# Prepare data for LLRQ K matrix estimation
def prepare_llrq_data(df, metabolite_cols):
    """Convert concentration time series to format for LLRQ fitting"""

    # Filter valid time points (post-pulse)
    post_pulse = df[df.time_s >= 0].copy()

    # Extract concentration matrix (time × metabolites)
    concentrations = post_pulse[metabolite_cols].values
    times = post_pulse.time_s.values

    # Handle missing values (interpolate or drop)
    mask = ~np.isnan(concentrations).any(axis=1)
    concentrations = concentrations[mask]
    times = times[mask]

    return times, concentrations

# Example: Prepare nucleotide data for LLRQ
nucleotide_cols = ['ATP_mean', 'ADP_mean', 'AMP_mean']
times, concs = prepare_llrq_data(pulse1, nucleotide_cols)
print(f"Prepared {len(times)} time points for {len(nucleotide_cols)} nucleotides")
```

## Applications for LLRQ Package

This time series data is well-suited for fitting K matrices in the LLRQ (Log-Linear Reaction Quotient) framework:

### 1. **Metabolic Network Dynamics**
- **Nucleotides**: Model ATP/ADP/AMP ratios and energy charge dynamics
- **Glycolysis metabolites**: Analyze pathway flux and regulation
- **TCA metabolites**: Study central carbon metabolism

### 2. **K Matrix Estimation Workflow**
```python
# Pseudo-code for LLRQ fitting
from llrq import ReactionNetwork, LLRQDynamics
from llrq.estimation.k_estimation import KMatrixEstimator

# 1. Define reaction network (e.g., simplified glycolysis)
network = ReactionNetwork(species=['G6P', 'F6P', 'F1,6P2'],
                         reactions=['PGI', 'PFK'])

# 2. Extract concentration data
pulse_data = metabolites[metabolites.dataset == 'pulse_1']
c_matrix = pulse_data[['G6P_mean', 'F6P_mean', 'F1,6P2_mean']].values

# 3. Fit K matrix using physical constraints
estimator = KMatrixEstimator(network, equilibrium_concentrations)
K_fitted = estimator.fit_timeseries(times, c_matrix)
```

### 3. **Multi-scale Integration**
- **Gene expression → Enzyme levels → Kinetic parameters**
- **Metabolite ratios → Reaction quotients → Thermodynamic forces**
- **Cross-validation across pulse replicates**

### 4. **Control Applications**
- **Perturbation response**: Glucose pulse as step input
- **Frequency analysis**: Oscillatory dynamics in metabolites
- **Robustness**: Consistency across experimental replicates

## Citation and Attribution

If you use this data, please cite the original publication (details to be confirmed from source):
- Molecular Systems Biology publication msb4100083
- Data processing and CSV conversion by LLRQ package tools

## Technical Notes

- **Conversion date**: Generated using Python pandas with xlrd library
- **Missing value handling**: Preserved as NaN for user-defined imputation
- **Precision**: Float64 for all numerical values
- **Encoding**: UTF-8 CSV format for cross-platform compatibility

---

For questions about data processing or LLRQ applications, see the main package documentation.
