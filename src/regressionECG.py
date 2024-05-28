# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
#
# # Step 1: Load the Data
# data = pd.read_csv("data/08215_features.csv")  # Replace "your_data.csv" with the path to your CSV file
#
# # Step 2: Prepare the Data
# # Exclude unnecessary columns
# features = data[["hrv_sdnn", "hrv_rmssd"]]
# target = data["num_AFIB_annotations"].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary: 1 if AFIB annotations > 0, else 0
#
# # Step 3: Split Data into Training and Testing Sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
# # Step 4: Train the Model
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # Step 5: Evaluate the Model
# train_accuracy = model.score(X_train, y_train)
# test_accuracy = model.score(X_test, y_test)
# print("Training Accuracy:", train_accuracy)
# print("Testing Accuracy:", test_accuracy)
#
# # Step 6: Make Predictions
# predictions = model.predict(X_test)
#
# # Step 7: Evaluate Predictions
# accuracy = accuracy_score(y_test, predictions)
# report = classification_report(y_test, predictions)
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(report)
