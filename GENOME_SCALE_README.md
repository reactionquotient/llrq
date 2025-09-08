# Genome-Scale Model Support in LLRQ

The LLRQ package has been enhanced with comprehensive support for genome-scale metabolic models, including efficient handling of large SBML files like yeast-GEM, human-GEM, and other community models.

## Key Features

### 🚀 Performance Optimizations

- **Sparse Matrix Support**: Automatic detection and use of sparse matrices for >95% sparse networks
- **Memory Efficiency**: 700x memory reduction for yeast-GEM (from 88MB to 0.12MB)
- **Fast Loading**: Optimized SBML parsing with lazy loading options
- **Scalable Operations**: Efficient conservation law detection and matrix operations

### 🧬 FBC Extension Support

Complete support for the SBML Flux Balance Constraints (FBC) extension:
- **Flux Bounds**: Parse upper and lower bounds for all reactions
- **Optimization Objectives**: Extract objective functions for flux balance analysis
- **Gene-Protein-Reaction Associations**: Parse gene association rules
- **Gene Products**: Handle gene product definitions and associations

### 🏢 Compartment-Aware Analysis

- **Compartment Detection**: Automatic identification of cellular compartments
- **Transport Analysis**: Distinguish between internal and transport reactions
- **Submodel Extraction**: Create compartment-specific submodels
- **Cross-Compartment Statistics**: Analyze metabolite distribution across compartments

### 🔧 Specialized Tools

- **Numerical Stability Checks**: Detect potential numerical issues in large models
- **Model Statistics**: Comprehensive analysis of model structure and properties
- **Performance Monitoring**: Track loading and computation times
- **Pathway Extraction**: Create pathway-specific submodels

## Quick Start

### Loading a Genome-Scale Model

```python
from llrq import load_genome_scale_model

# Load yeast-GEM with automatic optimizations
analyzer = load_genome_scale_model("yeast-GEM.xml")

# Print comprehensive model analysis
analyzer.print_summary()
```

### Creating LLRQ Systems

```python
from llrq import from_sbml

# Automatically detects large models and uses optimizations
network, dynamics, solver, visualizer = from_sbml("yeast-GEM.xml")

print(f"Created LLRQ system: {network.n_species} species, {network.n_reactions} reactions")
print(f"Sparse format: {network.is_sparse}, Sparsity: {network.sparsity:.2%}")
```

### Analyzing Compartments

```python
# Get compartment breakdown
compartments = analyzer.get_compartment_analysis()

for comp_id, data in compartments.items():
    print(f"{comp_id}: {data['n_species']} species, {data['n_internal_reactions']} internal reactions")

# Extract cytoplasmic submodel
cyto_model = analyzer.extract_compartment_submodel('c')
```

### Checking Model Quality

```python
# Numerical stability analysis
stability = analyzer.check_numerical_stability()

if stability['stability_warnings']:
    for warning in stability['stability_warnings']:
        print(f"⚠ {warning}")
```

## Supported Models

The enhanced LLRQ package has been tested with:

- **yeast-GEM**: 2,806 species, 4,131 reactions (✅ Fully supported)
- **human-GEM**: Expected to work with similar performance
- **E. coli models**: iML1515, iJO1366 (✅ Expected compatibility)
- **Custom SBML models**: Any FBC-compliant SBML Level 3 model

## Performance Benchmarks

### yeast-GEM Performance (2,806 species, 4,131 reactions):

| Operation | Time | Memory |
|-----------|------|--------|
| Model Loading | 1.0s | 350MB peak |
| Sparse Matrix Creation | 0.008s | 0.12MB |
| LLRQ System Creation | 1.2s | - |
| Conservation Law Detection | 4.7s | - |
| Compartment Analysis | <0.1s | - |

### Memory Comparison:

- **Dense Matrix**: 88.44 MB
- **Sparse Matrix**: 0.12 MB
- **Reduction**: 700x smaller

## Architecture

### New Modules

1. **`genome_scale.py`**: Main analyzer with performance optimizations
2. **Enhanced `sbml_parser.py`**: FBC extension support and sparse matrix creation
3. **Enhanced `reaction_network.py`**: Sparse matrix operations
4. **Enhanced `solver.py`**: Sparse matrix compatibility

### Integration Points

```python
# Automatic optimization selection
from llrq import from_sbml

# For large models (>1MB SBML files):
# - Uses GenomeScaleAnalyzer
# - Creates sparse matrices automatically
# - Provides performance warnings
network, dynamics, solver, viz = from_sbml("large_model.xml")

# For small models:
# - Uses standard parsing
# - Dense matrices for efficiency
# - No performance overhead
```

## Compartment Codes

Common compartment abbreviations in genome-scale models:

- `c`: Cytoplasm
- `m`: Mitochondria
- `n`: Nucleus
- `e`: Extracellular space
- `er`: Endoplasmic reticulum
- `g`: Golgi apparatus
- `p`: Peroxisome
- `v`: Vacuole
- `mm`: Mitochondrial membrane
- `lp`: Lipid particle

## API Reference

### GenomeScaleAnalyzer

```python
class GenomeScaleAnalyzer:
    def __init__(self, sbml_file: str, lazy_load: bool = True)

    def get_model_statistics(self) -> Dict[str, Any]
    def create_network(self, use_sparse: bool = None) -> ReactionNetwork
    def extract_compartment_submodel(self, compartment_ids: Union[str, List[str]]) -> 'GenomeScaleAnalyzer'
    def extract_pathway_submodel(self, reaction_ids: List[str], include_connected: bool = True) -> 'GenomeScaleAnalyzer'
    def check_numerical_stability(self) -> Dict[str, Any]
    def get_compartment_analysis(self) -> Dict[str, Dict[str, Any]]
    def print_summary(self)
```

### Enhanced SBMLParser

```python
class SBMLParser:
    def get_fbc_objectives(self) -> Dict[str, Any]
    def get_gene_products(self) -> Dict[str, Dict[str, Any]]
    def create_stoichiometric_matrix(self, species_ids: List[str], reactions: List[Dict[str, Any]], use_sparse: bool = None)
```

### Enhanced ReactionNetwork

```python
class ReactionNetwork:
    @property
    def is_sparse(self) -> bool

    @property
    def sparsity(self) -> float

    def get_reactant_stoichiometry_matrix(self) -> Union[np.ndarray, sparse.spmatrix]
    def get_product_stoichiometry_matrix(self) -> Union[np.ndarray, sparse.spmatrix]
```

## Examples

See `examples/genome_scale_demo.py` for a comprehensive demonstration of all genome-scale features using the yeast-GEM model.

## Testing

Run the genome-scale test suite:

```bash
python -m pytest tests/test_genome_scale.py -v
```

## Future Enhancements

Planned improvements for genome-scale support:

- **Parallel Processing**: Multi-core support for large matrix operations
- **Streaming Parsing**: Handle models too large to fit in memory
- **Model Validation**: Enhanced SBML validation and repair
- **Integration with COBRApy**: Seamless interoperability with flux balance analysis
- **GPU Acceleration**: CUDA support for matrix operations

## Contributing

When contributing genome-scale features:

1. Test with multiple model sizes (small, medium, genome-scale)
2. Ensure backward compatibility with existing small models
3. Include memory and performance benchmarks
4. Add comprehensive test coverage
5. Document any new compartment handling or FBC features

---

For questions about genome-scale functionality, please refer to the main documentation or open an issue on GitHub.
