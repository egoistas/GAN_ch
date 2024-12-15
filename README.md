
# Generative Adversarial Networks (GAN)

## Overview
This repository implements a **Generative Adversarial Network (GAN)** designed to generate synthetic data or images. GANs are a class of neural networks where two models, a generator and a discriminator (critic), are trained simultaneously through adversarial learning. The generator creates realistic samples, and the discriminator evaluates their authenticity, pushing both networks to improve iteratively.

### Key Features
- **Customizable Architecture:** Configurable generator and discriminator models.
- **Pre-trained Models:** Includes `generator.pth` and `critic.pth`.
- **Extensive Training Pipeline:** Tools for training, loss visualization, and saving models.
- **Utility Scripts:** Scripts for pre-processing data, managing checkpoints, and evaluating model performance.

---

## Repository Structure
```plaintext
GAN_ch/
├── config.py         # Configuration file for hyperparameters and model settings
├── critic.pth        # Pre-trained critic model
├── generator.pth     # Pre-trained generator model
├── model.py          # GAN model architecture
├── train.py          # Training script
├── utils.py          # Utility functions for training and evaluation
```

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Installation
Clone the repository:
```bash
git clone https://github.com/egoistas/GAN_ch.git
cd GAN_ch
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Training the GAN
Use the `train.py` script to train the GAN model. You can configure hyperparameters in `config.py`.

### Example
```bash
python train.py --epochs 50 --batch_size 128 --lr 0.0002
```

#### Parameters
- **`epochs`**: Number of training epochs.
- **`batch_size`**: Number of samples per batch.
- **`lr`**: Learning rate for both the generator and the critic.

---

## Pre-trained Models
### Generator
The generator is responsible for creating synthetic data that mimics the real dataset.

#### Usage
```python
from model import Generator
import torch

generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

noise = torch.randn(1, 100)  # Latent vector
fake_data = generator(noise)
```

### Critic
The critic evaluates the authenticity of generated data.

#### Usage
```python
from model import Critic
import torch

critic = Critic()
critic.load_state_dict(torch.load("critic.pth"))
critic.eval()

output = critic(fake_data)
```

---

## Visualizations

### Training Progress
During training, losses for both the generator and critic are logged and can be visualized.

#### Example Plot:
```python
import matplotlib.pyplot as plt

# Sample loss data
epochs = range(1, 51)
g_losses = [0.9, 0.8, ..., 0.1]  # Example generator loss
c_losses = [1.2, 1.1, ..., 0.5]  # Example critic loss

plt.plot(epochs, g_losses, label="Generator Loss")
plt.plot(epochs, c_losses, label="Critic Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Progress")
plt.show()
```

![Loss Visualization](./assets/loss_plot.png)

### Generated Samples
After training, the generator can produce synthetic samples. Below is an example:

![Generated Samples](./assets/generated_samples.png)

---

## Use Cases
1. **Synthetic Data Generation:** Create realistic data for machine learning models when data is limited.
2. **Image Generation:** Generate images from latent noise vectors.
3. **Data Augmentation:** Enhance datasets with synthetic samples to improve model generalization.

---

## Limitations
- Training GANs can be unstable without careful tuning of hyperparameters.
- Requires significant computational resources for large datasets.
- Generated samples may not perfectly match the real data distribution.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or collaboration, contact me at [georgischristides@gmail.com](mailto:georgischristides@gmail.com).
