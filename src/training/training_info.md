---
title: "Training Info"
order: 2
chapter: 1
section: 2
layout: "md.jlmd"
tags: []
---

# Training_12Jul2025.jl - SymAE Earthquake Training Pipeline

## Overview

This Pluto notebook implements a comprehensive training and analysis pipeline for a **Symmetric Autoencoder (SymAE)** model designed to extract coherent signals from earthquake seismogram data.

## Hardware & Environment

- **GPU Acceleration**: CUDA-enabled (GPU device 1)
- **Deep Learning Frameworks**: Flux.jl, Enzyme.jl for automatic differentiation
- **Computation Backend**: cuDNN for optimized neural network operations

## Model Architecture

### SymAE Configuration
```julia
symae_parameters = (
    nt = 192,      # Number of time samples
    p = 30,        # First latent dimension
    q = 30,        # Second latent dimension  
    k = 1          # Kernel size parameter
)
```

### Key Components
- **Convolutional Autoencoder**: Deep architecture for seismogram encoding/decoding
- **Spatial Transformer Networks**: Learns optimal time shifts to align waveforms
- **Coherence Module**: Extracts common source signatures across multiple recordings

## Data Processing

### Input Data
- **Source**: Earthquake seismogram database via `EqDataLoad.jl`
- **Format**: Preprocessed waveforms transferred to GPU memory
- **Windowing**: Extracts samples 9-200, resulting in 192-sample sequences
- **Normalization**: Applied to ensure consistent amplitude scaling

### Optional Processing
- **Envelope Extraction**: Toggle-able Hilbert transform for amplitude envelopes
- **Healpix Binning**: Spatial organization of earthquakes on spherical grid

## Training Strategy

### Multi-Stage Learning
1. **Stage 1**: 50 epochs, learning rate = 0.001
2. **Stage 2**: 50 epochs, learning rate = 0.0001  
3. **Stage 3**: 100 epochs, learning rate = 0.00001

### Loss Function
- **Reconstruction Loss**: MSE between input and decoded seismograms
- **Coherence Penalty**: Weighted term (γ=100) encouraging aligned signals
- **Temperature Parameter**: 0.25 for spatial transformer softmax annealing

### Training Parameters
```julia
Training_Para(
    nepoch = 100,
    gamma = 1f2,              # Coherence weight
    initial_learning_rate = 0.00001,
    temperature = 0.25f0       # Transformer temperature
)
```

## Key Functionality

### 1. Coherent Signal Extraction
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

### 2. Waveform Alignment
- Learns per-trace time shifts to maximize coherence
- Applies learned shifts to generate aligned ideal seismograms
- Tracks optimization loss history

### 3. Spatial Analysis
- Groups earthquakes by Healpix pixel location
- Analyzes coherent signals within spatial bins
- Color-codes results by geographic region

## Visualization Features

### Interactive Plots
- **Loss History**: Training convergence monitoring
- **Source Time Functions**: Stacked waveform displays with color coding
- **Reconstructions**: Side-by-side comparison of input vs. output
- **Envelope Comparison**: Mean displacement envelopes

### Plot Functions
```julia
plot_stf(stf_array, names, scale, title="")  # STF visualization
plot_loss_history(loss_history)              # Training diagnostics
```

## Data Selection & Analysis

### Interactive UI Elements
- **Earthquake Selector**: Choose specific events from database
- **Reload Network Button**: Reinitialize model architecture
- **Envelope Toggle**: Switch between raw waveforms and envelopes
- **Station Name Display**: Shows recording metadata

### Analysis Workflow
1. Select earthquake event from dropdown
2. Load corresponding recordings (up to `kidx_max_rec` traces)
3. Extract coherent signal using SymAE
4. Visualize results with interactive plots
5. Compare with mean displacement envelopes

## Model Persistence

### Saving Trained Models
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