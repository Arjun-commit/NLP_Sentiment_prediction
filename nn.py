import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
import numpy as np
import os

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def f1_metric(y_true, y_pred):
    """Calculates the F1 score."""
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)


def train(weights_path="model_weights.h5", train_path="train.csv", dev_path="dev.csv",
          num_hidden_layers=2, dropout_rate=0.3, learning_rate=2e-5, epochs=8, batch_size=16):
    #load datasets
    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})

    #extract label column names
    labels = hf_dataset["train"].column_names[1:]

    def process_labels(example):
        """Process labels into a multi-label vector."""
        return {"labels": [float(example[label]) for label in labels]}

    #tokenize and process labels
    hf_dataset = hf_dataset.map(process_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    #convert datasets to TensorFlow format
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        shuffle=True,
        batch_size=batch_size
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=batch_size
    )

    #define the DistilRoBERTa model
    distilroberta = transformers.TFAutoModel.from_pretrained("distilroberta-base")

    input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")

    #extract embeddings
    embeddings = distilroberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    #add multiple dense layers for better representation
    x = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    #output layer for multi-label classification
    output = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)

    #build and compile the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(multi_label=True), f1_metric]
    )

    #train the model
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        ]
    )

    #save the model weights
    model.save_weights(weights_path)
def predict(weights_path="model_weights.h5", test_path="test.csv"):
    #ensure weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{weights_path}' not found. Train the model first.")

    #rebuild the model architecture
    distilroberta = transformers.TFAutoModel.from_pretrained("distilroberta-base")
    input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")

    #model architecture
    embeddings = distilroberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    x = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    for _ in range(2):  # Use same hyperparameters as in training
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(7, activation="sigmoid")(x)  # Adjust units for your labels

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    #load the saved weights
    model.load_weights(weights_path)

    #load test data
    test_df = pd.read_csv(test_path)

    #tokenize input data
    hf_dataset = datasets.Dataset.from_pandas(test_df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    #convert dataset to TensorFlow format
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16
    )

    #generate predictions
    predictions = model.predict(tf_dataset)

    #assign predictions to corresponding columns
    test_df.iloc[:, 1:] = np.where(predictions > 0.5, 1, 0)

    #save predictions to a zipped CSV file
    test_df.to_csv("submission.zip", index=False, compression=dict(method="zip", archive_name="submission.csv"))


if __name__ == "__main__":
    # Manually set arguments for Jupyter Notebook environment
    args = argparse.Namespace()
    args.command = "train"  # or "predict"
    args.train_path = "train.csv"
    args.dev_path = "dev.csv"
    args.test_path = "test.csv"
    args.weights_path = "model_weights.h5"
    args.num_hidden_layers = 2
    args.dropout_rate = 0.3
    args.learning_rate = 2e-5
    args.epochs = 8
    args.batch_size = 16

    if args.command == "train":
        train(weights_path=args.weights_path, train_path=args.train_path, dev_path=args.dev_path,
              num_hidden_layers=args.num_hidden_layers, dropout_rate=args.dropout_rate,
              learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "predict":
        predict(weights_path=args.weights_path, test_path=args.test_path)
