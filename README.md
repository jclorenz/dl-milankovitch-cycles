# Milankovitch Cycle Detection in Synthetic Series Using Deep Learning

This repository provides the implementation of three deep learning architectures for identifying Milankovitch cycles in synthetic sedimentary time series: (1) Transformer, (2) Fourier Neural Operator (FNO), and (3) a proposed hybrid model combining spectral and attention-based representations. The models perform multi-label classification of four orbital components: precession (~21 kyr), obliquity (~41 kyr), short eccentricity (~100 kyr), and long eccentricity (~405 kyr), under varying sedimentation rates and noise conditions. The repository includes code and configurations for dataset generation, model training, evaluation, and experimental reproducibility.

## Repository Structure

The repository is organized as follows:

```text
milankovitch-deep-learning/
├── astronomical_solutions/       Astronomical solution files used for dataset generation
├── 01_dataset_generation.ipynb   Synthetic dataset generation
├── 02_transformer_only.ipynb     Transformer-only model training
├── 03_fno_only.ipynb             FNO-only model training
├── 04_hybrid.ipynb               Hybrid model training
├── 05_evaluation.ipynb           Model evaluation and architecture comparison
├── README.md                     Project documentation and usage instructions
├── LICENSE                       MIT license
└── .gitignore                    Excluded generated files and directories
```


## Dataset

The synthetic dataset was generated from the ZB18a astronomical solution (Zeebe & Lourens, 2019), covering the interval from 0 to 50 Ma. It contains 25,760 sedimentary time series, each comprising 2,000 samples, with individual and combined orbital components: precession (~21 kyr), obliquity (~41 kyr), short eccentricity (~100 kyr), and long eccentricity (~405 kyr).

To simulate different sedimentary conditions, the dataset incorporates:

- **Sedimentation rates:** 1–10 cm/kyr, with constant, trend, and step variations.
- **Noise:** combinations of red (AR1) and white noise at three intensity levels (n1, n2, and n3), alongside noise-free series.
- **Orbital components:** individual cycles and selected combinations of two or three cycles.
- **Preprocessing:** z-score normalization applied individually to each series.

The dataset was divided into training (70%), validation (15%), and testing (15%) subsets using a temporal separation strategy to reduce overlap between astronomical intervals.

The dataset generation notebook (`01_dataset_generation.ipynb`) and the astronomical solution files used in this study are available in the repository root and the `astronomical_solutions/` directory, respectively.

### Astronomical solutions

The astronomical data used in this study were derived from the ZB18a astronomical solution (Zeebe and Lourens, 2019). Climatic precession and obliquity were calculated using the snvecR package (Zeebe and Lourens, 2022; Kocken and Zeebe, 2023), adopting present-day values for Earth's dynamical ellipticity and tidal dissipation. The resulting astronomical series are provided in this repository and were used for the generation of synthetic sedimentary records.

To isolate the short (E1, ~100 kyr) and long (E2, ~405 kyr) eccentricity components, a Butterworth band-pass filter was applied to the ZB18a astronomical solution using Acycle software (Li et al., 2019). The frequency bands were set to 0.0068–0.0120 cycles/kyr for E1 and 0.00215–0.0027 cycles/kyr for E2, respectively.

References:
### References

- Kocken, I., & Zeebe, R. E. (2023). *snvecR: Calculate Earth's Obliquity and Precession in the Past*. Zenodo. https://doi.org/10.5281/zenodo.7865279

- Li, M., Hinnov, L., & Kump, L. (2019). Acycle: Time-series analysis software for paleoclimate research and education. *Computers & Geosciences*, 127, 12–22. https://doi.org/10.1016/j.cageo.2019.02.011

- Zeebe, R. E., & Lourens, L. J. (2019). Solar System chaos and the Paleocene–Eocene boundary age constrained by geology and astronomy. *Science*, 365, 926–929. https://doi.org/10.1126/science.aax0612

- Zeebe, R. E., & Lourens, L. J. (2022). A deep-time dating tool for paleo-applications utilizing obliquity and precession cycles: The role of dynamical ellipticity and tidal dissipation. *Paleoceanography and Paleoclimatology*, 37, e2021PA004349. https://doi.org/10.1029/2021PA004349

### Dataset Generator

To generate the synthetic dataset:

1. Open `01_dataset_generation.ipynb` in your preferred Jupyter-compatible environment (e.g., JupyterLab, Jupyter Notebook, or VS Code).
2. Ensure that the required astronomical solution files are available in the `astronomical_solutions/` directory.
3. Run all cells sequentially (*Run All*).
4. The generated dataset and metadata will be saved to the `synthetic_dataset/` directory.

The generated dataset is not included in the repository and must be created before training the models.

## Models

This repository provides implementations of three deep learning architectures for multi-label classification of Milankovitch cycles in synthetic sedimentary time series:

- **Transformer:** An attention-based architecture that processes input sequences through patch embedding and Transformer encoder layers to capture temporal relationships.
- **Fourier Neural Operator (FNO):** A spectral architecture that uses Fourier layers to capture frequency-domain representations of the input sequences.
- **Hybrid model:** The proposed architecture combines parallel FNO and Transformer branches, integrating spectral and attention-based representations for multi-label classification.

All models receive univariate sedimentary time series containing 2,000 samples and produce four output logits corresponding to precession, obliquity, short eccentricity (E1), and long eccentricity (E2). These logits are subsequently converted into probabilities using the sigmoid function.

### Model Training

Before training the models, the synthetic dataset must be generated following the instructions provided in the Dataset Generator section.

To train the models:

1. Open the corresponding notebook from the repository root:
   - `02_transformer_only.ipynb` — Transformer-only model.
   - `03_fno_only.ipynb` — FNO-only model.
   - `04_hybrid.ipynb` — Hybrid model.
2. Verify the dataset paths and training configurations specified in the notebook.
3. Run all cells sequentially (*Run All*).

The models were trained using the AdamW optimizer and binary cross-entropy loss with logits (BCEWithLogitsLoss). Training was performed with a maximum of 30 epochs and early stopping based on validation macro F1. Five independent runs were conducted for each architecture using random seeds 0–4.

Model checkpoints and prediction files are generated locally during training. The prediction files are saved in the `synthetic_dataset/predictions/` directory and are subsequently used for model evaluation.

For additional details on model architectures, hyperparameters, and training procedures, refer to the accompanying manuscript.

## Model Evaluation

The evaluation procedures are provided in `05_evaluation.ipynb`, located in the repository root.

To evaluate the trained models:

1. Execute the training notebooks for all three architectures to generate the required prediction files.
2. Open `05_evaluation.ipynb` in a Jupyter-compatible environment.
3. Ensure that the prediction files (`.npz`) are available in the `synthetic_dataset/predictions/` directory.
4. Run all cells sequentially (*Run All*).

Model performance was assessed using macro F1, AUC-ROC, and subset accuracy. Classification thresholds were determined individually for each orbital component using a Neyman–Pearson approach, with a target false-positive rate of 5% based on the pure-noise validation subset.

The evaluation includes comparisons between architectures, noise conditions, sedimentation rate variations, and orbital component combinations.

## Dependencies and Computational Requirements

### Software Dependencies

The models were implemented in Python using PyTorch. The main software dependencies include:

- Python 3.12.13
- PyTorch 2.6.0 (CUDA 12.4)
- NumPy 2.4.6
- SciPy 1.17.1
- scikit-learn 1.8.0
- Matplotlib 3.10.9

The notebooks can be executed in a Jupyter-compatible environment.

### Computational Requirements

The experiments were conducted using the following computational configuration:

- **Operating system:** Windows 11 (build 26100)
- **Processor (CPU):** Intel Core i7-9700K @ 3.60 GHz
- **System memory (RAM):** 16 GB
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)

GPU acceleration is recommended for reproducing the training experiments.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

## Contact
- **Author:** Júlia Carolina Lorenz
- **Institution:** Graduate Program in Applied Computing (Unisinos)
- **Email:** ju.julorenz@gmail.com
