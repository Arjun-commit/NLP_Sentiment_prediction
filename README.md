# NLP SENTIMENT ANALYSIS AND PREDICTION MODEL

This project implements a multi-label text classification model using a pre-trained DistilRoBERTa model with TensorFlow. It provides functionalities to train the model on custom datasets and to generate predictions on new, unseen data.

# Features
1. DistilRoBERTa Integration: Leverages the power of the DistilRoBERTa transformer for robust text embeddings.

2. Multi-label Classification: Designed to handle tasks where each input can belong to multiple categories simultaneously.

3. Customizable Training: Allows for configuration of hyperparameters like the number of hidden layers, dropout rate, learning rate, epochs, and batch size.

4. F1 Score Metric: Includes a custom F1 score calculation for evaluating multi-label classification performance.

5. Early Stopping: Implements early stopping during training to prevent overfitting.

6. Prediction Generation: Generates predictions on test data and saves them in a zipped CSV format suitable for submission.

# Getting Started
## Prerequisites
```1. Python 3.8 or higher```
```2. pip package manager```

## Installation
Clone the repository:

```git clone https://github.com/your-username/your-repository-name.git```
```cd your-repository-name```


### Create a virtual environment (recommended):

```python -m venv venv```
```source venv/bin/activate  # On Windows, use `venv\Scripts\activate```

## Install the required packages:

```pip install -r requirements.txt```

## Data Preparation

The project expects CSV files for training, development (validation), and testing. Each CSV file should have a text column containing the input text and subsequent columns representing the labels. For training and development data, these label columns should contain binary values (0 or 1).

- train.csv: Training data.

- dev.csv: Development (validation) data.

- test.csv: Test data (for prediction).


## Usage
The script can be run from the command line or within a Jupyter Notebook environment (as demonstrated in the provided if __name__ == "__main__": block).

### Training the Model
To train the model, use the train command. You can customize various hyperparameters.

```
python your_script_name.py train --train_path train.csv --dev_path dev.csv --weights_path model_weights.h5 \
--num_hidden_layers 2 --dropout_rate 0.3 --learning_rate 2e-5 --epochs 8 --batch_size 16
```
### Arguments:

```--train_path``` : Path to the training CSV file (default: train.csv).

```--dev_path``` : Path to the development CSV file (default: dev.csv).

```--weights_path``` : Path to save the trained model weights (default: model_weights.h5).

```--num_hidden_layers``` : Number of dense hidden layers to add after the RoBERTa embeddings (default: 2).

```--dropout_rate``` : Dropout rate for the dense layers (default: 0.3).

```--learning_rate``` : Learning rate for the Adam optimizer (default: 2e-5).

```--epochs``` : Number of training epochs (default: 8).

```--batch_size``` : Batch size for training and validation (default: 16).

### Making Predictions

To make predictions on new data, use the predict command. Ensure that the weights_path points to a trained model's weights.

```
python your_script_name.py predict --test_path test.csv --weights_path model_weights.h5
```

### Arguments:

```--test_path``` : Path to the test CSV file (default: test.csv).

```--weights_path``` : Path to the trained model weights file (default: model_weights.h5).

The predictions will be saved to a zipped CSV file named submission.zip in the current directory. The submission.zip file will contain a submission.csv file inside it, with the predicted binary labels (0 or 1) for each input text.

## Code Structure
```tokenizer``` : Global tokenizer initialized from ```distilroberta-base```.

```tokenize(examples)``` : Function to tokenize text data, converting it to input_ids and attention_mask.

```f1_metric(y_true, y_pred)``` : Custom TensorFlow function to calculate the F1 score for multi-label classification.

```train(...)``` : Main function for training the model. It loads datasets, processes labels, builds the DistilRoBERTa-based model with custom dense layers, compiles, trains, and saves the weights.

```predict(...)``` : Main function for generating predictions. It rebuilds the model, loads saved weights, tokenizes test data, and outputs predictions to a zipped CSV.

```if __name__ == "__main__":``` : Entry point for the script, handling command-line arguments and calling train or predict accordingly. It also includes a manual argument setup for Jupyter Notebook compatibility.

## Model Architecture

The model utilizes a pre-trained distilroberta-base model for generating contextualized embeddings. These embeddings are then fed into a GlobalAveragePooling1D layer, followed by a series of dense layers with ReLU activation and dropout for regularization. The final output layer uses a sigmoid activation function to produce probabilities for each label, suitable for multi-label classification.

## Evaluation

During training, the model is evaluated using Binary Crossentropy loss, AUC (Area Under the Curve) for multi-label classification, and the custom F1 score metric. Early stopping is employed to monitor val_loss and stop training if it doesn't improve for a specified number of epochs.

## Customization

```Hyperparameters``` : Experiment with num_hidden_layers, dropout_rate, learning_rate, epochs, and batch_size to optimize performance for your specific dataset.

```Model Architecture``` : The dense layers after the RoBERTa embeddings can be further customized (e.g., changing the number of units, adding more complex layers) if needed.

```Tokenizer Settings``` : Adjust max_length and padding in the tokenize function based on your text data characteristics.
