import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('party_data.csv')

#Preprocessing
encoded_data = pd.get_dummies(data, drop_first=True).astype(int)

#print("\nOne-Hot Encoded Data:")
#print(encoded_data)
#print(data.columns.to_list)

X = encoded_data.drop('ok guest_ok', axis=1)  # Features
y = encoded_data['ok guest_ok']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# KNN - Test for different values of k
k_values = [3, 5, 11, 17]

for k in k_values:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"\nK = {k}")
    print(f"Train Accuracy: {accuracy_train:.2f}")
    print(f"Test Accuracy: {accuracy_test:.2f}")

    # Kalkulerer confusion matrix og precision for 'ok' guests
    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:")
    print(cm)
    print("Precision for 'ok' guests:", cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0)

    # Print overfitting difference
    print(f"Overfitting Difference (Train - Test): {accuracy_train - accuracy_test:.2f}")

# Logistisk regresjon med standard regularisering (penalty='l2')
log_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear')
log_reg_l2.fit(X_train, y_train)

y_pred_train_l2 = log_reg_l2.predict(X_train)
y_pred_test_l2 = log_reg_l2.predict(X_test)

accuracy_train_l2 = accuracy_score(y_train, y_pred_train_l2)
accuracy_test_l2 = accuracy_score(y_test, y_pred_test_l2)

# Evaluer modellen med regularisering
print("\nLogistisk Regresjon med penalty='l2'")
print(f"Train Accuracy: {accuracy_train_l2:.2f}")
print(f"Test Accuracy: {accuracy_test_l2:.2f}")

cm_l2 = confusion_matrix(y_test, y_pred_test_l2)
print("Confusion Matrix:")
print(cm_l2)
print("Precision for 'ok' guests:", cm_l2[1, 1] / (cm_l2[1, 1] + cm_l2[0, 1]) if (cm_l2[1, 1] + cm_l2[0, 1]) > 0 else 0)
print(f"Overfitting Difference (Train - Test): {accuracy_train_l2 - accuracy_test_l2:.2f}")

# Logistisk regresjon uten regularisering (penalty='none')
log_reg_none = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
log_reg_none.fit(X_train, y_train)

y_pred_train_none = log_reg_none.predict(X_train)
y_pred_test_none = log_reg_none.predict(X_test)

accuracy_train_none = accuracy_score(y_train, y_pred_train_none)
accuracy_test_none = accuracy_score(y_test, y_pred_test_none)

# Evaluer modellen uten regularisering
print("\nLogistisk Regresjon med penalty='none'")
print(f"Train Accuracy: {accuracy_train_none:.2f}")
print(f"Test Accuracy: {accuracy_test_none:.2f}")

cm_none = confusion_matrix(y_test, y_pred_test_none)
print("Confusion Matrix:")
print(cm_none)
print("Precision for 'ok' guests:", cm_none[1, 1] / (cm_none[1, 1] + cm_none[0, 1]) if (cm_none[1, 1] + cm_none[0, 1]) > 0 else 0)
print(f"Overfitting Difference (Train - Test): {accuracy_train_none - accuracy_test_none:.2f}")

# Beslutningstre med Gini-kriteriet
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)

y_pred_train_gini = dt_gini.predict(X_train)
y_pred_test_gini = dt_gini.predict(X_test)

accuracy_train_gini = accuracy_score(y_train, y_pred_train_gini)
accuracy_test_gini = accuracy_score(y_test, y_pred_test_gini)

# Evaluer modellen med Gini-kriteriet
print("\nBeslutningstre med Gini-kriteriet")
print(f"Train Accuracy: {accuracy_train_gini:.2f}")
print(f"Test Accuracy: {accuracy_test_gini:.2f}")

cm_gini = confusion_matrix(y_test, y_pred_test_gini)
print("Confusion Matrix:")
print(cm_gini)
print("Precision for 'ok' guests:", cm_gini[1, 1] / (cm_gini[1, 1] + cm_gini[0, 1]) if (cm_gini[1, 1] + cm_gini[0, 1]) > 0 else 0)
print(f"Overfitting Difference (Train - Test): {accuracy_train_gini - accuracy_test_gini:.2f}")

# Beslutningstre med Entropy-kriteriet
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

y_pred_train_entropy = dt_entropy.predict(X_train)
y_pred_test_entropy = dt_entropy.predict(X_test)

accuracy_train_entropy = accuracy_score(y_train, y_pred_train_entropy)
accuracy_test_entropy = accuracy_score(y_test, y_pred_test_entropy)

# Evaluer modellen med Entropy-kriteriet
print("\nBeslutningstre med Entropy-kriteriet")
print(f"Train Accuracy: {accuracy_train_entropy:.2f}")
print(f"Test Accuracy: {accuracy_test_entropy:.2f}")

cm_entropy = confusion_matrix(y_test, y_pred_test_entropy)
print("Confusion Matrix:")
print(cm_entropy)
print("Precision for 'ok' guests:", cm_entropy[1, 1] / (cm_entropy[1, 1] + cm_entropy[0, 1]) if (cm_entropy[1, 1] + cm_entropy[0, 1]) > 0 else 0)
print(f"Overfitting Difference (Train - Test): {accuracy_train_entropy - accuracy_test_entropy:.2f}")