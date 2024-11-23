# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec



"""
Load Glove Embeddings
Step 1 of the Assignment
"""

def load_glove_embeddings(glove_input_file, word2vec_output_file):
    """
    Convert GloVe format to Word2Vec format and load embeddings.
    """
    # Convert GloVe format to Word2Vec format
    glove2word2vec(glove_input_file, word2vec_output_file)
    
    # Load Word2Vec format embeddings using Gensim
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return word_vectors

# Paths to GloVe files
glove_input_file = "glove.6B/glove.6B.50d.txt"
word2vec_output_file = "glove.6B/glove.6B.50d.word2vec.txt"

# Load the embeddings
word_vectors = load_glove_embeddings(glove_input_file, word2vec_output_file)



"""""
3 - Loading and Vectorizing the Data
Step 2 of the Assignment
"""""
# Load the datasets
train_data = pd.read_csv("Dataset/train.csv")
test_data = pd.read_csv("Dataset/test.csv")
dev_data = pd.read_csv("Dataset/dev.csv")

# Default zero vector for missing words
default_vector = np.zeros(50)

def process_review(review, word_vectors):
    """
    Converts a single review into a 50-dimensional vector.
    - Retrieve embeddings for each word in the review.
    - Use the mean of the embeddings to represent the review.
    """
    word_embeddings = []
    for word in review.split():
        if word in word_vectors:
            word_embeddings.append(word_vectors[word])
        else:
            word_embeddings.append(default_vector)
    
    # Compute the mean vector for the review
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return default_vector

# Apply process_review to each review in the dataset, converting all reviews into their corresponding 50-dimensional vectors.
def process_reviews(data, word_vectors):
    review_vectors = [process_review(review, word_vectors) for review in data['review']]
    return np.array(review_vectors)


# Prepare data and map sentiments to binary values
X_train = process_reviews(train_data, word_vectors)
train_data['sentiment'] = train_data['sentiment'].map({'positive': 1, 'negative': 0})
y_train = train_data['sentiment'].values

X_test = process_reviews(test_data, word_vectors)
test_data['sentiment'] = test_data['sentiment'].map({'positive': 1, 'negative': 0})
y_test = test_data['sentiment'].values


X_val = process_reviews(dev_data, word_vectors)
dev_data['sentiment'] = dev_data['sentiment'].map({'positive': 1, 'negative': 0})
y_val = dev_data['sentiment'].values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")




"""""
Normalization
Step 3 of the Assignment
"""""
def normalize_features(X_train: np.ndarray, X_val: np.ndarray = None, X_test: np.ndarray = None):
    """
    Normalize features using min-max normalization.
    Calculate min and max only from the training set.
    """
    # Calculate min and max values from training data
    min_values, max_values = X_train.min(axis=0), X_train.max(axis=0)
    
    # Normalize training data
    X_train_normalized = (X_train - min_values) / (max_values - min_values)
    
    # Normalize validation and test data using training min/max
    X_val_normalized = (X_val - min_values) / (max_values - min_values) if X_val is not None else None
    X_test_normalized = (X_test - min_values) / (max_values - min_values) if X_test is not None else None
    
    return X_train_normalized, X_val_normalized, X_test_normalized


# Normalize the datasets
X_train_normalized, X_val_normalized, X_test_normalized = normalize_features(X_train, X_val, X_test)




"""""
Logistic Regression
Step 4 of the Assignment
"""""

def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))

def compute_loss_and_gradients(X, y, weights, bias):
    """Compute the binary cross-entropy loss and gradients."""
    m = X.shape[0]  # Number of samples
    
    # Compute predictions
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    
    # Compute loss
    loss = -(1 / m) * np.sum(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    
    # Compute gradients
    dw = (1 / m) * np.dot(X.T, (predictions - y))
    db = (1 / m) * np.sum(predictions - y)
    
    return loss, dw, db

def mini_batch_gradient_descent_with_tracking(X: np.ndarray, y: np.ndarray, 
                                              X_val: np.ndarray, y_val: np.ndarray,
                                              learning_rate: float = 0.01, 
                                              epochs: int = 1000, batch_size: int = 16):
    """Train logistic regression using mini-batch gradient descent with accuracy and loss tracking."""
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = X[indices], y[indices]

        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            # Compute gradients and update parameters
            loss, dw, db = compute_loss_and_gradients(X_batch, y_batch, weights, bias)
            weights -= learning_rate * dw
            bias -= learning_rate * db

        # Evaluate the full training set accuracy
        train_predictions = sigmoid(np.dot(X, weights) + bias) >= 0.5
        train_accuracy = np.mean(train_predictions == y)
        train_accuracies.append(train_accuracy)

        # Compute the full training loss
        train_loss, _, _ = compute_loss_and_gradients(X, y, weights, bias)
        train_losses.append(train_loss)

        # Evaluate the validation set accuracy
        val_predictions = sigmoid(np.dot(X_val, weights) + bias) >= 0.5
        val_accuracy = np.mean(val_predictions == y_val)
        val_accuracies.append(val_accuracy)

        # Compute the validation loss
        val_loss, _, _ = compute_loss_and_gradients(X_val, y_val, weights, bias)
        val_losses.append(val_loss)

        # Print every 10 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
                  f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    return weights, bias, train_accuracies, val_accuracies, train_losses, val_losses



# Validation and Hyperparameter Tuning
def evaluate_model(X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, 
                   learning_rates: list, epochs: int, batch_sizes: list):
    """
    Evaluate logistic regression model for different learning rates and batch sizes.
    """
    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Testing learning_rate={lr}, batch_size={batch_size}")

            # Train the model using mini-batch gradient descent
            weights, bias, train_accuracies, val_accuracies, _, _ = mini_batch_gradient_descent_with_tracking(
                X_train, y_train, X_val, y_val, learning_rate=lr, epochs=epochs, batch_size=batch_size
            )

            # Compute validation accuracy
            val_predictions = sigmoid(np.dot(X_val, weights) + bias) >= 0.5
            val_accuracy = np.mean(val_predictions == y_val)

            # Store the result
            results.append({"learning_rate": lr, "batch_size": batch_size, "accuracy": val_accuracy})
            print(f"Validation Accuracy: {val_accuracy:.4f}")

    return results

# Hyperparameter tuning
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]
epochs = 1000

# Evaluate models with different hyperparameters
validation_results = evaluate_model(
    X_train_normalized, y_train, X_val_normalized, y_val, 
    learning_rates, epochs, batch_sizes
)

# Find the best hyperparameters
best_model = max(validation_results, key=lambda x: x['accuracy'])
print("Best Hyperparameters:", best_model)




# Final Evaluation on the Test Dataset
def final_evaluation_on_test(X_train, y_train, X_val, y_val, X_test, y_test, 
                             learning_rates, epochs, batch_sizes):
    """
    Perform final evaluation on the test dataset after tuning hyperparameters on validation data.
    """
    best_model = None
    best_weights, best_bias = None, None
    best_accuracy = 0

    # Tune hyperparameters on validation data
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Testing learning_rate={lr}, batch_size={batch_size}")
            weights, bias, train_accuracies, val_accuracies, _, _ = mini_batch_gradient_descent_with_tracking(
                X_train, y_train, X_val, y_val, learning_rate=lr, epochs=epochs, batch_size=batch_size
            )
            
            # Compute validation accuracy
            val_predictions = sigmoid(np.dot(X_val, weights) + bias) >= 0.5
            val_accuracy = np.mean(val_predictions == y_val)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            # Save the best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = {"learning_rate": lr, "batch_size": batch_size, "accuracy": val_accuracy}
                best_weights, best_bias = weights, bias

    print("\nBest Hyperparameters:", best_model)

    # Final test evaluation
    test_predictions = sigmoid(np.dot(X_test, best_weights) + best_bias) >= 0.5
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")

    return best_model, best_weights, best_bias, test_accuracy


# Hyperparameters and training configuration
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]
epochs = 1000

# Perform hyperparameter tuning and test evaluation
best_model, best_weights, best_bias, test_accuracy = final_evaluation_on_test(
    X_train_normalized, y_train, X_val_normalized, y_val, 
    X_test_normalized, y_test, learning_rates, epochs, batch_sizes
)


