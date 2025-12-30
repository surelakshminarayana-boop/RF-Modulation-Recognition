***

# RF Modulation Recognition

A deep learning–based system for automatic RF modulation classification from raw complex IQ samples.  
The project provides a Streamlit web app for real‑time signal upload, modulation prediction, and visualization (constellation, spectrogram, etc.).[1][2]

***

## Features

- **End‑to‑end pipeline** from raw IQ data to predicted modulation labels (e.g., BPSK, QPSK, 8PSK, QAM16, AM, FM).[2]
- **CNN‑LSTM model** that learns features directly from IQ sequences for robust performance across SNR levels.[2]
- **Streamlit UI** for simple file upload, one‑click inference, and interactive plots.[1][2]
- **Modular codebase** with separate components for data loading, preprocessing, training, inference, and visualization.[2]
- **Exported model weights** in `.h5` format for edge/production deployment.[2]

***

## Project Structure

```bash
RF-Modulation-Recognition/
├── app.py               # Streamlit web app for inference and visualization
├── models/              # Trained model weights and model definitions
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── LICENSE              # MIT license
└── .gitignore
```

- `app.py`: Loads trained model weights, handles file uploads, runs inference, and generates plots.[1][2]
- `models/`: Contains CNN‑LSTM architecture code and saved `.h5` weights.[2]
- `requirements.txt`: Lists all Python packages required to run the app.[1]

***

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/surelakshminarayana-boop/RF-Modulation-Recognition.git
   cd RF-Modulation-Recognition
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install Streamlit, TensorFlow/Keras, NumPy, and other required libraries.[1][2]

***

## Usage

### 1. Running the Streamlit App

```bash
streamlit run app.py
```

- Open the local URL shown in the terminal (usually `http://localhost:8501`).  
- Use the UI to:
  - Upload RF IQ data files (`.pkl`/`.npy` as supported by the app).[2]
  - Click the **Identify Signal** button to run the model.[2]
  - View:
    - Predicted modulation type with class probabilities (e.g., “8PSK – 95% confidence”).[2]
    - Constellation diagram and other diagnostic plots.[2]

### 2. Supported Input Format

- Raw complex IQ samples, preprocessed into standardized arrays (shape and format must match the training setup).[2]
- Typical input is derived from public RF modulation datasets (e.g., RadioML‑style `.npy` files).[2]

***

## Model Overview

- **Architecture**:  
  - CNN layers for spatial/constellation feature extraction from IQ samples.  
  - LSTM layers for temporal modeling of sequence dynamics.[2]
  - Final dense + softmax layer for modulation classification.[2]

- **Training pipeline**:
  - DataLoader normalizes and splits the raw dataset into train/validation sets.[2]
  - `ModelBuilder` constructs and compiles the CNN‑LSTM model with configured hyperparameters.[2]
  - `Trainer` handles training, evaluation, and saving `.h5` weights.[2]

- **Target use‑cases**:
  - Cognitive radio and dynamic spectrum access.  
  - 5G/IoT networks, military communications, and interference avoidance at the edge.[2]

***

## Configuration

Core configuration (paths, batch size, epochs, class labels, etc.) is centralized in a `Config` component (e.g., `Config.DATAPATH`, `Config.BATCHSIZE`, `Config.EPOCHS`, `Config.CLASSES`).[2]

You can adjust:

- Dataset path and file names.  
- Number of epochs, batch size, learning rate.  
- List of modulation classes to train/predict.[2]

***

## Roadmap / Future Work

- Add more modulation types and higher‑order QAM schemes.[2]
- Improve robustness at very low SNR and in fading channels.[2]
- Provide ready‑to‑use example datasets and notebooks for training.[2]
- Containerize the app (Docker) for easier deployment.  

***

## License

This project is released under the **MIT License**. See the [`LICENSE`](./LICENSE) file for details.[1]

***
