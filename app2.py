import streamlit as st
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim import AdamW, SGD  # Correct import for SGD and AdamW


# Function to train the ELECTRA model
def train_electra_model(train_texts, train_labels, num_epochs, optimizer_choice):
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator",
                                                             num_labels=len(label_encoder.classes_))

    encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_encodings, val_encodings, train_labels_split, val_labels_split = train_test_split(
        encodings["input_ids"], train_labels, test_size=0.2, random_state=42
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset({'input_ids': train_encodings}, train_labels_split)
    val_dataset = TextDataset({'input_ids': val_encodings}, val_labels_split)

    st.write(f"Training dataset size: {len(train_dataset)}")
    st.write(f"Validation dataset size: {len(val_dataset)}")

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    class ProgressCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            progress = state.epoch / num_epochs
            progress_bar.progress(progress)
            status_text.text(f"Training Progress: {progress * 100:.2f}%")

    if optimizer_choice == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=5e-5)
    elif optimizer_choice == 'SGD':
        optimizer = SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()],
        optimizers=(optimizer, None)
    )

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    eval_results = trainer.evaluate()

    trainer.model.eval()
    with torch.no_grad():
        val_inputs = val_encodings.to(trainer.model.device)
        logits = trainer.model(val_inputs)["logits"]
        preds = np.argmax(logits.cpu().numpy(), axis=1)

    return eval_results.get("eval_accuracy", None), preds, val_labels_split


# Streamlit UI
st.title("META")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    columns_to_drop = st.multiselect("Select columns to drop", data.columns.tolist())

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        st.write("Updated Data Preview after dropping columns:")
        st.dataframe(data.head())

    # Visualization part to select both X and Y columns
    st.subheader("Visualize Data")
    x_column = st.selectbox("Select feature column for X-axis", data.columns)
    y_column = st.selectbox("Select target column for Y-axis", [col for col in data.columns if col != x_column])

    # Display the visualization
    if x_column and y_column:
        st.write(f"Scatter plot of {x_column} vs {y_column}")

        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(data[x_column]) and pd.api.types.is_numeric_dtype(data[y_column]):
            ax.scatter(data[x_column], data[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f"{x_column} vs {y_column}")
        elif pd.api.types.is_numeric_dtype(data[y_column]):
            data.groupby(x_column)[y_column].mean().plot(kind='bar', ax=ax)
            ax.set_ylabel(f"Average of {y_column}")
        else:
            st.write("Selected columns are not suitable for plotting.")

        st.pyplot(fig)

    target_column = st.selectbox("Select target column for training", data.columns)
    num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=3)
    num_rows = st.number_input("Number of Rows for Training", min_value=1, max_value=data.shape[0], value=100)
    optimizer_choice = st.selectbox("Select optimizer", ['AdamW', 'SGD'])

    st.write(f"Dataset size: {data.shape[0]} rows")

    if st.button("Train Model"):
        if data.shape[0] > num_rows:
            data = data.sample(n=num_rows, random_state=42)

        train_texts = data.drop(columns=[target_column]).astype(str).agg(' '.join, axis=1).tolist()
        train_labels = data[target_column].tolist()

        with st.spinner("Training the model..."):
            final_accuracy, preds, actuals = train_electra_model(train_texts, train_labels, num_epochs,
                                                                 optimizer_choice)
            st.success("Model trained successfully!")
            st.write(
                f"Final Accuracy: {final_accuracy:.2f}" if final_accuracy is not None else "Accuracy not available.")

            st.write("Predicted vs Actual Values:")
            prediction_df = pd.DataFrame({'Actual': actuals, 'Predicted': preds})
            prediction_df['Index'] = prediction_df.index
            fig, ax = plt.subplots()
            ax.plot(prediction_df['Index'], prediction_df['Actual'], label='Actual', marker='o')
            ax.plot(prediction_df['Index'], prediction_df['Predicted'], label='Predicted', marker='x')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Class Label')
            ax.set_title('Predicted vs Actual Values')
            ax.legend()
            st.pyplot(fig)

    if st.button("Download Model"):
        import shutil

        shutil.make_archive('model', 'zip', './model')
        with open('model.zip', 'rb') as f:
            st.download_button('Download Trained Model', f, file_name='model.zip', mime='application/zip')
