# Model Files

This directory contains large SBML model files used for testing and demonstrating the genome-scale capabilities of the LLRQ package.

## Git LFS Usage

Large model files (*.xml) in this directory are tracked using **Git Large File Storage (Git LFS)** to keep the main repository lightweight while still maintaining version control.

### Files Tracked by Git LFS:

- `yeast-GEM.xml` - Saccharomyces cerevisiae genome-scale metabolic model (11.4 MB)
  - 2,806 species, 4,131 reactions, 14 compartments
  - Used to demonstrate LLRQ performance with large-scale models
  - Source: [yeast-GEM repository](https://github.com/SysBioChalmers/yeast-GEM)

### For Developers:

If you're working with this repository and encounter Git LFS files, you may need to:

1. **Install Git LFS** (if not already installed):
   ```bash
   # macOS with Homebrew
   brew install git-lfs

   # Or download from https://git-lfs.github.com/
   ```

2. **Initialize Git LFS** for the repository:
   ```bash
   git lfs install
   ```

3. **Download LFS files** (if cloning):
   ```bash
   git lfs pull
   ```

### File Information:

- **yeast-GEM.xml**: Community consensus genome-scale model of yeast metabolism
  - Format: SBML Level 3 with FBC extension
  - Compartments: cytoplasm (c), mitochondria (m), nucleus (n), extracellular (e), etc.
  - Features: Flux bounds, gene associations, optimization objectives
  - Perfect for testing genome-scale LLRQ functionality

### Why These Files Are Important:

These large model files are essential for:

1. **Performance Testing**: Validating that LLRQ can handle realistic genome-scale models
2. **Feature Demonstration**: Showcasing FBC extension support, compartment analysis, etc.
3. **Benchmarking**: Measuring memory efficiency improvements with sparse matrices
4. **User Examples**: Providing realistic examples for documentation and tutorials

The files are legitimate scientific data, not bloat, and are essential for the proper functioning and demonstration of the genome-scale features in LLRQ.
