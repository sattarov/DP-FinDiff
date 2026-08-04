# DP-FinDiff

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sattarov/DP-FinDiff/blob/master/examples/compact.ipynb)

Implementation of the paper: **"Privacy Preserving Diffusion Models for Mixed-Type Tabular Data Generation"** [link](https://arxiv.org/abs/2512.00638).

DP-FinDiff is an extension of FinDiff that integrates **Differential Privacy (DP)** for generating privacy-preserving synthetic tabular data. It is a PyTorch-based framework specifically designed to handle the complexities of financial datasets (e.g., mixtures of continuous and categorical variables). It leverages diffusion models with swappable neural network backbones to synthesize realistic records while providing formal privacy guarantees.

## Features

- **Differential Privacy**: Easy-to-use integration with Opacus to enable training with differential privacy for provable privacy guarantees.
- **Efficient Categorical Handling**: Learnable embeddings for categorical data, with support for decoding via distance-based matching or direct logit prediction.
- **Conditional Generation**: Support for label-conditioned synthetic data generation using Classifier-Free Guidance.
- **Multiple Neural Backbones**: Choose between MLP, Transformer, and U-Net backbones for the diffusion process.
- **Robust Data Transformation**: Built-in `DataTransformer` handles standardizing numerical features (Standard, MinMax, Robust, Power, Quantile) and encoding categorical features.
- **Customizable Diffusion Schedulers**: Configurable noise schedules including linear, quadratic, sigmoid, and exponential.

## Examples

You can explore DP-FinDiff's capabilities using our example notebooks on Google Colab:

- [Compact Example](./examples/compact.ipynb): A quick-start guide to get you up and running (examples/compact.ipynb).
  

## Architecture

The framework is highly modular and broken down into four core components:

1. **The Orchestrator (`findiff/model.py`)**
   - **`FinDiff`**: The main user-facing class. It acts as the central orchestrator that wires together data transformations, diffusion mathematics, and the neural network. It handles the complete lifecycle, including the training loop (`fit()`), synthetic data generation (`sample()`), and the integration of differential privacy via Opacus.

2. **Data Processing Pipeline (`findiff/data.py`)**
   - **`DataTransformer`**: A robust preprocessing pipeline built specifically for tabular data. It scales numerical features and encodes categorical variables. Importantly, it manages the `inverse_transform` to accurately convert raw diffusion outputs back into a readable pandas DataFrame.
   - **`FinDiffDataset`**: A standard PyTorch dataset wrapper to feed transformed tensors into dataloaders.

3. **Neural Network Backbones (`findiff/backbones.py`)**
   - **`FinDiffSynthesizer`**: The core predictive model that takes numerical data, embedded categorical data, and time-step embeddings to predict the noise during the reverse diffusion step.
   - **Interchangeable Architectures**: The framework implements three separate architectures that can be swapped based on data complexity:
     - `MLPBackbone`: Uses residual blocks for standard, lightweight tabular generation.
     - `TransformerBackbone`: An attention-based tabular model.
     - `UNetBackbone`: Adapts U-Net principles (typically used in image generation) for tabular representations.

4. **Diffusion Mathematics (`findiff/diffusion.py`)**
   - **`BaseDiffuser`**: Handles the underlying math for the diffusion process. It manages noise schedulers (Linear, Quadratic, Sigmoid, Exponential) and computes the parameters required to gradually add noise during the forward pass and iteratively denoise during generation.

## Usage Example

Below is a minimal example demonstrating how to train a DP-FinDiff model and generate privacy-preserving synthetic tabular data.

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from findiff.data import DataTransformer, FinDiffDataset
from findiff.model import FinDiff

# 1. Prepare and transform your data
df = pd.DataFrame({
    "cat_col": ["A", "B", "A", "C"],
    "num_col": [1.5, 2.3, 0.9, 3.1]
})

transformer = DataTransformer(
    categorical_cols=["cat_col"], 
    numerical_cols=["num_col"],
    numerical_scaler="standard"
)
transformed_data = transformer.fit_transform(df)

# 2. Create a DataLoader
dataset = FinDiffDataset(
    cat_dataset=torch.tensor(transformed_data.get("cat")),
    num_dataset=torch.tensor(transformed_data.get("num"), dtype=torch.float32)
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Initialize and train the DP-FinDiff model
# To enable Differential Privacy, provide the dp_params dictionary.
# To train without DP, simply omit the dp_params argument.
dp_parameters = {
    "target_epsilon": 10.0,
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,
}

model = FinDiff(
    data_transformer=transformer,
    backbone_type="mlp",  # Try 'transformer' or 'unet'
    diffusion_total_steps=1000,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dp_params=dp_parameters, # Omit for non-private training
    train_dataloader=dataloader # Required for DP
)

model.fit(dataloader)

# 4. Generate synthetic data
synthetic_df = model.sample(n_samples=5)
print(synthetic_df)
```

## Citation

If this project helped your research, please consider giving it a star and citing it in your work. It helps keep the project alive.

```bibtex
@article{sattarov2025privacy,
  title={Privacy Preserving Diffusion Models for Mixed-Type Tabular Data Generation},
  author={Sattarov, Timur and Schreyer, Marco and Borth, Damian},
  journal={arXiv preprint arXiv:2512.00638},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
