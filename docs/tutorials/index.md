# Tutorials

Step-by-step tutorials for common QME workflows and use cases.

## Getting Started Tutorials

### [Basic Optimization](basic_optimization.md)
Learn the fundamentals of molecular optimization with QME.
- Setting up your first calculation
- Understanding output files
- Choosing the right backend

### [Transition State Finding](transition_states.md)
Master transition state searches and pathway analysis.
- Local TS optimization from guess structures
- Two-ended TS searches between reactants and products
- Using NEB for reaction pathways

## Intermediate Tutorials

### [Frequency Analysis](frequency_analysis.md)
Calculate vibrational frequencies and thermodynamic properties.
- Computing Hessian matrices
- Identifying true minima vs saddle points
- Thermodynamic property calculations

### [Batch Processing](batch_processing.md)
Efficiently process multiple structures and workflows.
- Automating calculations for multiple molecules
- Parallel processing strategies
- Managing large datasets

## Advanced Tutorials

### [Custom Workflows](custom_workflows.md)
Build sophisticated computational chemistry workflows.
- Combining optimization with analysis
- Multi-step reaction pathway studies
- Integration with other computational tools

### [Performance Optimization](performance_optimization.md)
Get the most out of QME for large-scale studies.
- Backend selection for different systems
- GPU acceleration strategies
- Memory management and optimization

## Tutorial Format

Each tutorial includes:

- **Learning objectives** - What you'll accomplish
- **Prerequisites** - Required knowledge and setup
- **Step-by-step instructions** - Detailed walkthrough
- **Code examples** - Copy-paste ready commands
- **Expected output** - What results to expect
- **Troubleshooting** - Common issues and solutions
- **Next steps** - How to build on the tutorial

## Running the Tutorials

### Prerequisites

Before starting the tutorials, ensure you have:

1. **QME installed** with at least one backend:
   ```bash
   pip install qme-ml-ml[aimnet2]  # Recommended for beginners
   ```

2. **Example files** available:
   ```bash
   # Download tutorial files
   git clone https://github.com/rlaplaza-lab/qme.git
   cd qme/examples/example_files/
   ```

3. **Working environment**:
   ```bash
   # Test your installation
   qme --help
   qme opt --help
   ```

### Tutorial Files

Example structures used in tutorials:

- `benzene.xyz` - Simple aromatic molecule for basic optimization
- `reaction_001_reactant.xyz` - Reactant structure for TS searches
- `reaction_001_product.xyz` - Product structure for TS searches
- `reaction_001_ts.xyz` - Initial TS guess structure
- `batch_molecules/` - Directory with multiple structures

### Estimated Time

- **Basic tutorials**: 15-30 minutes each
- **Intermediate tutorials**: 30-60 minutes each
- **Advanced tutorials**: 1-2 hours each

## Support and Feedback

### Getting Help

If you encounter issues with the tutorials:

1. **Check prerequisites** - Ensure your environment is set up correctly
2. **Review troubleshooting sections** - Each tutorial includes common issues
3. **Search documentation** - Use the search function to find relevant info
4. **Ask for help** - Use GitHub Discussions for questions

### Improving Tutorials

We welcome feedback on tutorials:

- **Report issues** - If instructions don't work as expected
- **Suggest improvements** - Ideas for clarity or additional content
- **Contribute examples** - Share your own use cases
- **Request topics** - What tutorials would be helpful?

## Tutorial Roadmap

Planned future tutorials:

- **Constraint-based optimization** - Using geometric constraints
- **Solvent effects** - Including implicit solvation
- **Conformational analysis** - Systematic conformer searches
- **Reaction network analysis** - Multi-step reaction mechanisms
- **Integration with experiment** - Comparing with experimental data

## Related Resources

### Documentation

- **[User Guide](../user_guide/index.md)** - Comprehensive reference
- **[API Reference](../developer_guide/api_reference.md)** - Detailed API docs
- **[Examples](../benchmarks/examples.md)** - Additional code examples

### External Resources

- **ASE Documentation** - [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
- **SELLA Optimizer** - [https://github.com/zadorlab/sella](https://github.com/zadorlab/sella)
- **Machine Learning Potentials** - Background reading on ML methods

## Quick Reference

### Common Commands
```bash
# Basic optimization
qme opt molecule.xyz

# Different backends
qme opt molecule.xyz --backend aimnet2
qme opt molecule.xyz --backend mace --device cuda

# Transition state optimization
qme tsopt ts_guess.xyz

# Two-ended optimization
qme opt reactant.xyz --product product.xyz

# Help and options
qme opt --help
```

### Python Quick Start
```python
import qme

# Basic optimization
explorer = qme.Explorer.from_file("molecule.xyz")
result = explorer.run(mode="minima")

# Save results
explorer.save_structure(result['optimized_atoms'], "optimized.xyz")
```

Ready to get started? Begin with [Basic Optimization](basic_optimization.md)!
