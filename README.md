# Sentiment Analysis with GloVe Embeddings and Logistic Regression

This project builds a sentiment analysis model using GloVe word embeddings and logistic regression. It follows a structured pipeline that includes data processing, feature extraction, normalization, training, and evaluation. The notebook provides clear step-by-step instructions, and the implementation answers eight specific questions about model behavior and performance.

## Project Components

##### 1. GloVe Embeddings Loading
- Converts GloVe embeddings from .txt to Word2Vec format using Gensim.
- Loads the embeddings to be used for text vectorization.
##### 2. Data Preparation
- Processes train, validation (dev), and test datasets.
- Maps sentiment labels (positive, negative) to binary values (1, 0).
- Converts text reviews into 50-dimensional vectors by averaging word embeddings.
##### 3. Normalization
- Normalizes features using Min-Max normalization to improve gradient descent performance.
##### 4. Logistic Regression Implementation
- Implements logistic regression with mini-batch gradient descent.
- Tracks accuracy and loss for training and validation datasets during training.
##### 5. Hyperparameter Tuning
- Evaluates different learning rates and batch sizes.
- Identifies the optimal configuration based on validation accuracy.
##### 6. Final Model Evaluation
- Tests the model with the best hyperparameters on the test dataset.


## Questions Answered

#### Q1: Gradient Descent Convergence and Learning Rate Analysis (20 pts)
- Task: Experiment with learning rates (10000, 1000, 100, 10, 1, 0.01, 0.001, 0.0001, 0.00001).
- Analysis: Reports overflow cases, convergence behavior, training, and validation accuracy for each learning rate in a table.
- Additional Experiment: Extends the number of iterations to 10000 for 0.0001 and analyzes improvement.
#### Q2: Accuracy vs. Iterations (10 pts)
- Task: Plot training and validation accuracy over iterations for learning rates (100, 0.1, 0.001, 0.00001).
- Objective: Observe the effect of different learning rates on model convergence.
#### Q3: Loss vs. Iterations (10 pts)
- Task: Plot average training and validation loss over iterations for the same learning rates as in Q2.
- Objective: Compare accuracy trends to loss trends to understand model behavior.
#### Q4: Accuracy vs. Learning Rate (10 pts)
- Task: Plot validation accuracy for all learning rates that did not cause overflow.
- Objective: Identify the learning rate with the best validation performance.
#### Q5: Test Accuracy for Best Learning Rate (5 pts)
- Task: Evaluate the test dataset using the best learning rate identified in Q4.
#### Q6: Batch Size Optimization (10 pts)
- Task: Experiment with batch sizes (4, 8, 16, 32, 64).
- Analysis: Plot validation accuracy for each batch size to determine the best-performing size.
#### Q7: L2 Regularization (10 pts)
- Task: Add L2 regularization to the gradient descent implementation.
- Analysis: Evaluate the effect of regularization on training and validation accuracy for different λ values.
- Objective: Identify if regularization improves performance and recommend its usage.
#### Q8: Effect of Data Normalization (10 pts)
- Task: Train a model without normalization and compare results.
- Analysis: Examine training difficulty, required iterations, and best learning rate for unnormalized data.
- Objective: Plot training and validation accuracy for three learning rates and explain observations.

## How to Use

### Dependencies:
Ensure the following packages are installed:
</br>`numpy`, `pandas`, `matplotlib`, `gensim`
</br></br>Install via:

```bash
pip install numpy pandas matplotlib gensim
```

### Files and Directories:
`glove.6B.50d.txt`: Pre-trained GloVe embeddings.
`train.csv`, `dev.csv`, `test.csv`: Datasets for training, validation, and testing.

### Run the Notebook:
Open the Jupyter Notebook version for step-by-step explanations and interactive visualizations.
### Output:
- Tables, plots, and model performance for each question.
- Best hyperparameters and final test accuracy.
- Results Summary


Best learning rate: 0.01
</br>Best batch size: 16
</br>Test accuracy: 0.7875
