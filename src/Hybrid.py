import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class CNN_LSTM(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, lstm_units, dropout_rate):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(1, num_filters, kernel_size)
        self.batchnorm1 = nn.BatchNorm1d(num_filters)
        self.maxpool1 = nn.MaxPool1d(2, padding=1)  # Add padding
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.batchnorm2 = nn.BatchNorm1d(num_filters * 2)
        self.maxpool2 = nn.MaxPool1d(2, padding=1)  # Add padding
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(input_size=num_filters * 4, hidden_size=lstm_units,
                             num_layers=1, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units,
                             num_layers=1, batch_first=True, dropout=dropout_rate)
        self.lstm3 = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units,
                             num_layers=1, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("Input shape:", x.shape)  # Print the shape of the input
        x = self.conv1(x.permute(0, 2, 1))  # Adjust the dimension for the convolutional layer
        # print("Shape after unsqueeze:", x.shape)  # Print the shape after unsqueezing
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add a new dimension for the LSTM input
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Extract the last time step output
        x = self.softmax(x)
        return x


def prepare_data(df):
    # Filter out rows where SDNN > 500 ms
    df = df[df['hrv_sdnn'] <= 500]
    # Filter out rows where RMSSD > 500 ms
    df = df[df['hrv_rmssd'] <= 500]
    # Filter out rows where cv > 0.5 (50 % variability)
    df = df[df['cv'] <= 0.5]
    # Filter out rows where the signal_quality is lower than 0.3
    df = df[df['signal_quality'] >= 0.3]

    features = ['hrv_sdnn', 'hrv_rmssd', "hrv_mean", 'cv', "heart_rate_std", "heart_rate_mean", "sd1", "sd2"]
    x = df[features].values
    y = df['num_AFIB_annotations'].values  # Target: whether the patient has AFib

    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(x, y)

    # Reshape x_res to 3D tensor
    x_res = x_res.reshape((x_res.shape[0], x_res.shape[1], 1))

    return train_test_split(x_res, y_res, test_size=0.2, random_state=42)


def create_classification_report_image(class_report_df):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    cell_text = class_report_df.values
    table = plt.table(cellText=cell_text,
                      colLabels=class_report_df.columns,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.savefig("classification_report.png", bbox_inches='tight')
    plt.close()


def create_pdf(accuracy, conf_matrix):
    pdf_filename = "../reports/model_evaluation_CNN_LSTM_torch.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.drawImage("classification_report.png", 55, 250, width=500, preserveAspectRatio=True, mask='auto')
    c.drawString(270, height - 50, "Accuracy")
    c.drawString(242, height - 70, f"{accuracy}")
    # c.drawString(255, height - 100, "ROC AUC Score")
    # c.drawString(242, height - 120, f"{roc_auc}")
    c.drawString(245, height - 150, "Classification Report")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

    c.drawImage("confusion_matrix.png", 65, 0, width=500, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()


def main():
    df = pd.read_csv('../data/afdb_data.csv')
    x_train, x_test, y_train, y_test = prepare_data(df)

    # Print the shapes of x_train and x_test
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    # Convert NumPy arrays to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)  # Add a new dimension for the number of channels
    num_filters = 64
    kernel_size = 3
    lstm_units = 100
    dropout_rate = 0.3

    model = CNN_LSTM(input_shape, num_filters, kernel_size, lstm_units, dropout_rate)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy: {accuracy:.4f}')

        # Generate PDF report
        conf_matrix = confusion_matrix(y_test, predicted)
        class_report = classification_report(y_test, predicted, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df['labels'] = class_report_df.index
        cols = class_report_df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        class_report_df = class_report_df[cols]

        create_classification_report_image(class_report_df)
        create_pdf(accuracy, conf_matrix)


if __name__ == "__main__":
    main()
