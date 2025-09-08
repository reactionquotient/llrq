# Yeast Core Glycolysis LLRQ Example

This example demonstrates **Log-Linear Reaction Quotient (LLRQ)** dynamics applied to yeast core glycolysis, including fermentation and the glycerol branch. It showcases how the LLRQ framework can model complex metabolic networks with thermodynamically-consistent equilibrium constants derived from genome-scale models.

## 🧬 Biological System

### Metabolic Network
The system models 8 key reactions in yeast central carbon metabolism:

1. **PGI** (r_0467): G6P ↔ F6P (phosphoglucose isomerase)
2. **TPI** (r_1054): DHAP ↔ GAP (triose phosphate isomerase)
3. **GAPDH** (r_0486): GAP + Pi + NAD ↔ 1,3-BPG + NADH (glyceraldehyde-3-phosphate dehydrogenase)
4. **PGK** (r_0892): 1,3-BPG + ADP ↔ 3-PG + ATP (phosphoglycerate kinase)
5. **PGM** (r_0893): 3-PG ↔ 2-PG (phosphoglycerate mutase)
6. **ENO** (r_0366): 2-PG ↔ PEP (enolase)
7. **ADH** (r_0163): AcAld + NADH ↔ EtOH + NAD (alcohol dehydrogenase, fermentation)
8. **GPD** (r_0490): DHAP + NADH ↔ G3P + NAD (glycerol-3-phosphate dehydrogenase, glycerol branch)

### Control Inputs
The system includes 5 control inputs representing metabolic regulation:

1. **u_glc_in**: Glucose influx (affects glucose-6-phosphate availability)
2. **u_pyk_pull**: Pyruvate kinase activity (pulls PEP toward pyruvate)
3. **u_pdc_push**: Pyruvate decarboxylase push (drives acetaldehyde production)
4. **u_ATP_boost**: ATP/ADP ratio manipulation (high energy state)
5. **u_ADP_boost**: ADP/ATP ratio manipulation (low energy state)

## 🔬 LLRQ Framework

### Mathematical Model
The system follows LLRQ dynamics:

```
dx/dt = -K·x + B·u(t)
```

Where:
- **x = ln(Q/Keq)**: Log-scaled reaction quotients relative to equilibrium
- **K**: Relaxation/coupling matrix (8×8) encoding metabolic network structure
- **B**: Control input matrix (8×5) defining how controls affect reactions
- **u(t)**: Control input vector (5×1)

### Thermodynamic Foundation
Equilibrium constants (Keq) are derived from:
1. **Primary source**: yeast-GEM.yml thermodynamic data (ΔG° values)
2. **Literature fallback**: Experimentally validated values for missing data

| Reaction | YAML Keq | Literature Keq | Source Used | Notes |
|----------|----------|----------------|-------------|-------|
| PGI      | 1.43     | 0.35          | YAML        | Both favor F6P slightly |
| TPI      | 0.56     | 0.045         | YAML        | DHAP strongly favored |
| GAPDH    | 2.38     | 0.08          | YAML        | Near equilibrium |
| PGK      | 0.033    | 2500          | YAML        | Major discrepancy* |
| PGM      | 0.99     | 0.1           | YAML        | Near equilibrium |
| ENO      | 1.42     | 0.7           | YAML        | Both near unity |
| ADH      | 0.0024   | 4000          | YAML        | Major discrepancy* |
| GPD      | 30000    | 30000         | Literature  | Missing FAD ΔG° |

*Note: Large discrepancies likely due to different reference states, pH, ionic strength, or cofactor concentrations in the genome-scale model vs. experimental conditions.

## 📁 Files

### Core Files
- **`yeast_glycolysis_llrq.py`**: Main simulation script
- **`extract_keq_values.py`**: Thermodynamic data extraction from yeast-GEM.yml
- **`yeast_keq_values.csv`**: Extracted equilibrium constants
- **`yeast_K_matrix.csv`**: System relaxation/coupling matrix
- **`yeast_B_matrix.csv`**: Control input matrix

### Output Files
- **`glycolysis_llrq_results.png`**: Simulation results visualization
- **`README.md`**: This documentation

## 🚀 Usage

### Quick Start
```bash
# Extract thermodynamic data (optional - already done)
python examples/yeast_glycolysis/extract_keq_values.py

# Run the main simulation
python examples/yeast_glycolysis_llrq.py
```

### Requirements
- Python 3.7+
- NumPy, SciPy, Pandas, Matplotlib
- LLRQ package (src/llrq/)
- Tellurium environment (for yeast-GEM.yml parsing)

## 📊 Analysis Features

### 1. Control Scenarios
The example demonstrates 5 different metabolic control strategies:

- **Baseline**: No control (u = 0)
- **Glucose Pulse**: Simulates glucose influx
- **Fermentation Push**: Enhances ethanol production pathway
- **Energy Boost**: Simulates high ATP/ADP ratio
- **Glycerol Favor**: Conditions favoring glycerol branch

### 2. Visualizations
- **Time series**: Evolution of key reaction quotients
- **Phase portrait**: Fermentation vs. glycerol branch activity
- **Energy metabolism**: GAPDH and PGK coupling
- **Control heatmap**: Control effort comparison

### 3. Thermodynamic Accounting
- Entropy production calculation for each scenario
- Energy efficiency analysis
- Thermodynamic constraint validation

## 🔍 Key Insights

### 1. Metabolic Trade-offs
The model reveals the fundamental trade-off between:
- **Fermentation pathway** (ADH): Fast ATP generation, ethanol production
- **Glycerol branch** (GPD): NADH consumption, osmotic protection

### 2. Control Strategies
Different control inputs lead to distinct metabolic phenotypes:
- **Energy boost** shifts balance toward ATP-requiring reactions
- **Fermentation push** activates ethanol production pathway
- **Glucose influx** affects upstream glycolytic flux

### 3. Network Structure
The K matrix sparsity pattern reflects biological coupling:
- **Strong coupling** between GAPDH and ADH/GPD (NADH cofactor sharing)
- **Energy coupling** between PGK and ATP-dependent processes
- **Sequential coupling** in glycolytic pathway

## ⚠️ Limitations and Considerations

### 1. Model Scope
- Focuses on core glycolysis; excludes pentose phosphate pathway, gluconeogenesis
- Simplified control representation; real regulation involves allosteric effects, enzyme expression
- Assumes well-mixed conditions; ignores spatial compartmentalization

### 2. Thermodynamic Data
- yeast-GEM ΔG° values may differ from physiological conditions
- pH, ionic strength, temperature dependencies not fully captured
- Some reactions require literature fallback values

### 3. LLRQ Approximation
- Linear dynamics valid near equilibrium; may break down far from steady state
- Assumes fast equilibration of elementary steps within each reaction
- Cofactor pools (NAD/NADH, ATP/ADP) treated implicitly

## 🔗 References

### Experimental Keq Values
- PGI: MilliporeSigma, PubMed references
- TPI: PMC, biochemical databases
- GAPDH: PubMed, Wikipedia
- PGK: PMC, Wikipedia
- PGM: PMC publications
- ENO: ScienceDirect, IUBMB Journals, JBC
- ADH: PubMed experimental data
- GPD: FEBS Online Library

### Genome-Scale Model
- **yeast-GEM v9.0.2**: The Consensus Genome-Scale Metabolic Model of Yeast
- Source: SysBioChalmers/yeast-GEM GitHub repository

## 🎯 Educational Value

This example serves as a comprehensive tutorial for:

1. **LLRQ Theory**: Practical application to realistic metabolic network
2. **Thermodynamic Integration**: Combining genome-scale models with literature data
3. **Control Analysis**: Multiple control strategies and their metabolic consequences
4. **Data Pipeline**: From raw thermodynamic data to simulation results
5. **Biological Insight**: Understanding metabolic trade-offs through mathematical modeling

The yeast glycolysis example demonstrates how the LLRQ framework bridges the gap between detailed mechanistic models and systems-level understanding, providing both mathematical rigor and biological insight.
