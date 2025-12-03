# Earthquake-SymAE.jl

## Overview

This repository contains a **Symmetric Autoencoder (SymAE)** implementation for extracting coherent signals from earthquake seismogram data. The project uses deep learning to learn patterns in seismic waveforms while accounting for path effects through learned time shifts.

## Getting Started with Julia and Pluto Notebooks

### Installing Julia

1. **Download Julia**
   - Visit https://julialang.org/downloads/
   - Download Julia 1.9 or later (current stable release recommended)
   - Choose the installer for your operating system (Linux, macOS, or Windows)

2. **Install Julia**
   - Version 1.12.1

3. **Verify Installation**
   - Open a terminal and type `julia`
   - You should see the Julia REPL with a `julia>` prompt
   - Type `exit()` to quit

### Installing Pluto

1. **Launch Julia** from your terminal by typing `julia`

2. **Install Pluto** by entering package mode (press `]`) and typing:
   ```julia
   add Pluto
   ```

3. **Exit package mode** by pressing backspace

4. **Start Pluto** by running:
   ```julia
   import Pluto
   Pluto.run()
   ```

5. Your web browser should automatically open to `http://localhost:1234/`

### Opening the Training Notebook

1. **In the Pluto interface**, you'll see an option to "Open a notebook"

2. **Navigate to** or paste the path to the training notebook:
   ```
   src/training/Training_12Jul2025.jl
   ```

3. **Click "Open"** - Pluto will automatically install all required packages (this may take 10-15 minutes the first time)

4. Once loaded, the notebook is interactive! You can:
   - Run cells by clicking them or pressing Shift+Enter
   - Modify code and see results update automatically
   - Use the interactive UI elements (dropdowns, buttons, toggles)

## Training Pipeline: Training_12Jul2025.jl

### Model Architecture

#### SymAE Configuration
```julia
symae_parameters = (
    nt = 192,      # Number of time samples
    p = 30,        # First latent dimension
    q = 30,        # Second latent dimension  
    k = 1          # Kernel size parameter
)
```

#### Key Components
- **Convolutional Autoencoder**: Deep architecture for seismogram encoding/decoding
- **Spatial Transformer Networks**: Learns optimal time shifts to align waveforms
- **Coherence Module**: Extracts common source signatures across multiple recordings

### Data Processing

#### Input Data
- **Source**: Earthquake seismogram database via `EqDataLoad.jl`
- **Format**: Preprocessed waveforms transferred to GPU memory
- **Windowing**: Extracts samples 9-200, resulting in 192-sample sequences
- **Normalization**: Applied to ensure consistent amplitude scaling

#### Optional Processing
- **Envelope Extraction**: Toggle-able Hilbert transform for amplitude envelopes
- **Healpix Binning**: Spatial organization of earthquakes on spherical grid

### Training Strategy

#### Multi-Stage Learning
1. **Stage 1**: 50 epochs, learning rate = 0.001
2. **Stage 2**: 50 epochs, learning rate = 0.0001  
3. **Stage 3**: 100 epochs, learning rate = 0.00001

#### Loss Function
- **Reconstruction Loss**: MSE between input and decoded seismograms
- **Coherence Penalty**: Weighted term (γ=100) encouraging aligned signals

#### Training Parameters
```julia
Training_Para(
    nepoch = 100,
    gamma = 1f2,              # Coherence weight
    initial_learning_rate = 0.00001,
)
```

### Key Functionality

#### 1. Coherent Signal Extraction
**Function**: `get_coherent_information()`

Extracts optimal source time functions (STFs) from earthquake recordings by:
- Encoding seismograms into latent space
- Learning nuisance parameters (time shifts) to align traces
- Decoding aligned representations to extract coherent signals
- Optimizing over N iterations with specified learning rate

**Parameters**:
- `N=10`: Number of optimization iterations
- `nepochs=100`: Epochs per iteration
- `alpha=0.1`: Regularization weight
- `learning_rate=0.1`: Step size for nuisance optimization

#### 2. Waveform Alignment
- Learns per-trace time shifts to maximize coherence
- Applies learned shifts to generate aligned ideal seismograms
- Tracks optimization loss history

#### 3. Spatial Analysis
- Groups earthquakes by Healpix pixel location
- Analyzes coherent signals within spatial bins
- Color-codes results by geographic region

### Visualization Features

#### Interactive Plots
- **Loss History**: Training convergence monitoring
- **Source Time Functions**: Stacked waveform displays with color coding
- **Reconstructions**: Side-by-side comparison of input vs. output
- **Envelope Comparison**: Mean displacement envelopes

#### Plot Functions
```julia
plot_stf(stf_array, names, scale, title="")  # STF visualization
plot_loss_history(loss_history)              # Training diagnostics
```

### Data Selection & Analysis

#### Interactive UI Elements
- **Earthquake Selector**: Choose specific events from database
- **Reload Network Button**: Reinitialize model architecture
- **Envelope Toggle**: Switch between raw waveforms and envelopes
- **Station Name Display**: Shows recording metadata

#### Analysis Workflow
1. Select earthquake event from dropdown
2. Load corresponding recordings (up to `kidx_max_rec` traces)
3. Extract coherent signal using SymAE
4. Visualize results with interactive plots
5. Compare with mean displacement envelopes

### Model Persistence

#### Saving Trained Models
```julia
# Timestamp-based filenames
timestamp = now()
savename_para = savename(symae_parameters)

# Save model state
jldsave("SavedModels/syn_model-$(timestamp)-$(savename_para).jld2", 
        model_state = Flux.state(cpu(symae_model.model)))

# Save hyperparameters
jldsave("SavedModels/syn_para-$(timestamp)-$(savename_para).jld2"; 
        symae_parameters)

# Save training history
jldsave("SavedModels/syn_loss_history-$(timestamp)-$(savename_para).jld2"; 
        loss_history)
```

## Hardware & Environment

- **GPU Acceleration**: CUDA-enabled (GPU device 1)
- **Deep Learning Frameworks**: Flux.jl, Enzyme.jl for automatic differentiation
- **Computation Backend**: cuDNN for optimized neural network operations

### GPU Support Notes

Since the training pipeline uses CUDA for GPU acceleration:

1. **Check GPU availability**:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

2. **CUDA will be installed automatically** by Julia when you first run the notebook

3. **If you don't have a GPU**, modify the notebook:
   - Change `xpu = gpu` to `xpu = cpu`
   - Training will be slower but should still work

## Dependencies

### Core ML/DL
- Flux.jl, NNlib, OneHotArrays
- Optimisers.jl, ParameterSchedulers.jl
- MLUtils.jl for data handling

### Optimization
- BlackBoxOptim.jl, Metaheuristics.jl
- Enzyme.jl for automatic differentiation

### Signal Processing
- DSP.jl (FFT, filtering, Hilbert transform)
- FFTW.jl for fast Fourier transforms

### Visualization
- PlutoPlotly.jl for interactive plots
- PlotlyKaleido.jl for static image export
- ColorSchemes.jl for color mapping

### Data & I/O
- JLD2.jl for saving trained models
- HDF5.jl for data storage
- CSV.jl, DrWatson.jl for experiment management

## Output Products

1. **Trained Model Weights**: Saved as JLD2 files with timestamp
2. **Source Time Functions**: Extracted coherent signals per earthquake
3. **Aligned Seismograms**: Waveforms corrected for path delays
4. **Loss Curves**: Training diagnostics
5. **Publication Figures**: High-resolution plots (950×450, scale=8)

## Usage Notes

- Model reloading via `@use_memo` ensures fresh initialization
- GPU memory management handled automatically by CUDA.jl
- Interactive development enabled by Pluto's reactivity
- Memoization prevents redundant computations during notebook updates

## Troubleshooting

**Problem**: Packages fail to install
- **Solution**: Try running `] up` in the Julia REPL to update packages

**Problem**: CUDA errors
- **Solution**: Make sure NVIDIA drivers are installed and up to date

**Problem**: Pluto won't start
- **Solution**: Try a different port: `Pluto.run(port=1235)`

**Problem**: Out of memory errors
- **Solution**: Reduce batch size or use smaller data subsets

## Resources

- Julia Documentation: https://docs.julialang.org/
- Pluto Documentation: https://github.com/fonsp/Pluto.jl
- Flux.jl (Deep Learning): https://fluxml.ai/

## Repository Structure

```
src/
├── training/
│   ├── Training_12Jul2025.jl      # Main training notebook
│   ├── SymAE_architecture.jl      # Model architecture definitions
│   └── training_info.md           # Detailed training documentation
├── loading_data/
│   ├── EqDataLoad.jl              # Data loading utilities
│   └── deep_EQs_dataP.hdf5        # Earthquake dataset
└── _data/                         # Website metadata
```

## License

[Insert your license information here]

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.