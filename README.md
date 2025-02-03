# Upstage: Making AI Beneficial (0.966) Solution Source Code

This repository contains the source code for the **Upstage: Making AI Beneficial** competition solution, achieving a score of **0.966**. The solution leverages a transformer-based model (**SAINT**) for sequential learning tasks, with additional features like **pseudo-labeling** and **ensembling** for improved performance.

## Table of Contents
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Training](#training)
  - [Training Methods](#training-methods)
- [Inference](#inference)
- [Ensembling](#ensembling)
- [Weights and Reproducibility](#weights-and-reproducibility)
- [Acknowledgments](#acknowledgments)

## Requirements
To run the code, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Scikit-learn
- Joblib
- TQDM
- Termcolor

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Directory Structure
`sub_and_weights` directory contains the model weights and submission files generated during training and inference and the hyperparameter as images.

```
.
├── README.md
├── data
│   ├── sample_output.csv
│   ├── test.csv
│   └── train.csv
├── ensemble.py
├── requirements.txt
├── sub_and_weights
│   ├── subs
│   │   ├── ...
│   ├── v1
│   │   ├── ...
│   ├── v1_pseudo
│   │   ├── ...
│   ├── v2
│   │   ├── ...
│   ├── v2_pseudo
│   │   ├── ...
│   ├── v3
│   │   ├── ...
│   ├── v5
│   │   ├── ...
│   ├── v5_pseudo
│   │   ├── ...
│   ├── v6
│   │   ├── ...
│   ├── v6_pseudo
│   │   ├── ...
│   ├── v6_with_adam
│   │   ├── ...
│   └── v7
│       ├── ...
├── test.py
├── test.sh
├── train.py
└── train.sh
```

## Training
To train the model, use the following command:

```bash
bash train.sh <weights_dir> <data_dir>
```

### Arguments:
- `weights_dir`: Directory to save the model weights (**without** a trailing `/`).
- `data_dir`: Directory containing the competition data (**with** a trailing `/`).

#### Example:
```bash
bash train.sh ./sub_and_weights/weights ./data/
```

### Notes:
- The training script supports pseudo-labeling by providing the `--pseudo` argument.
- Training hyperparameters (e.g., epochs, batch size) can be adjusted in the script.

## Training Methods
The training process follows a structured pipeline involving:

1. **Data Preparation:**
   - The dataset is grouped by `student_id`, where sequences of interactions are extracted.
   - Features like `feature_1`, `feature_2`, `question_id`, `bundle_id`, and response correctness are preprocessed.
   - Pseudo-labeling is applied for additional training data if specified.

2. **Dataset and Dataloader:**
   - A custom `SAINTDataset` class is implemented to handle variable-length sequences and padding.
   - The dataset is divided into training and validation sets based on a predefined number of students.
   - `DataLoader` is used for efficient batch processing.

3. **Model Architecture:**
   - The model uses a **Transformer-based sequential encoder**.
   - **Feature embeddings** are used for categorical variables like `question_id`, `bundle_id`, `feature_3`, etc.
   - A **Feed-Forward Network (FFN)** and **LSTM layers** are incorporated to enhance sequence learning.
   - Dropout and layer normalization are used for better generalization.

4. **Training Loop:**
   - The model is trained using **CrossEntropyLoss**.
   - Optimization is handled by **Adam** or **SGD** (with an option for Adam specified in arguments).
   - **Cosine Annealing LR scheduler** is used for learning rate adjustments.
   - **KFold cross-validation** ensures robustness in training.

## Inference
To run inference and generate predictions, use the following command:

```bash
bash test.sh <weights_dir> <data_dir> <output_dir>
```

### Arguments:
- `weights_dir`: Directory containing the saved model weights (**without** a trailing `/`).
- `data_dir`: Directory containing the competition data (**with** a trailing `/`).
- `output_dir`: Directory to save the output submission files (**without** a trailing `/`).

#### Example:
```bash
bash test.sh ./sub_and_weights/weights ./data/ ./sub_and_weights/subs
```

## Ensembling
To ensemble multiple submission files, use the following command:

```bash
python3 ensemble.py --sub_dir <SUB_DIR> -d <DATA_DIR> -o <OUT_FILE>
```

### Arguments:
- `--sub_dir SUB_DIR`: Directory containing the submission files (**with** a trailing `/`).
- `-d DATA_DIR, --data_dir DATA_DIR`: Directory containing the competition data (**with** a trailing `/`).
- `-o OUT_FILE, --out_file OUT_FILE`: Output file name (**with `.csv` extension**).

#### Example:
```bash
python3 ensemble.py --sub_dir ./sub_and_weights/subs -d ./data/ -o final_output.csv
```

## Weights and Reproducibility
- The provided weights in `./sub_and_weights/v*/*.pth` are genuine and were used to achieve the final score.
- Due to the stochastic nature of training, reproducing the exact results may be challenging. It is recommended to use the provided weights for inference.

## Acknowledgments
- This solution is based on the **SAINT (Sequential Attention-based Interaction Network for Knowledge Tracing)** model.
- Special thanks to the **competition organizers** and the **open-source community** for providing valuable resources and tools.

Thank you for using this repository! If you have any questions or suggestions, feel free to open an issue or reach out.