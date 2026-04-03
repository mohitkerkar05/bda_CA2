# ----------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

df = pd.DataFrame(iris.data, columns=[
    "sepal_length", "sepal_width", "petal_length", "petal_width"
])
df['label'] = iris.target

print("Dataset loaded")


# ----------------------------------------
# STEP 2: Split Data
# ----------------------------------------
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)


# ----------------------------------------
# STEP 3: Distance Function
# ----------------------------------------
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))


# ----------------------------------------
# STEP 4: Distributed KNN (Chunk-based)
# ----------------------------------------
def knn_predict(test_point, train_data, k=3, chunk_size=20):
    
    distances = []
    
    # Simulate distributed chunks
    for i in range(0, len(train_data), chunk_size):
        chunk = train_data.iloc[i:i+chunk_size]
        
        for _, row in chunk.iterrows():
            dist = euclidean_distance(
                test_point[:-1].values,
                row[:-1].values
            )
            distances.append((dist, row['label']))
    
    # Sort distances
    distances.sort(key=lambda x: x[0])
    
    # Get top k
    neighbors = distances[:k]
    
    # Majority vote
    labels = [label for _, label in neighbors]
    return Counter(labels).most_common(1)[0][0]


# ----------------------------------------
# STEP 5: Test Model
# ----------------------------------------
correct = 0
total = len(test_df)

for _, row in test_df.iterrows():
    pred = knn_predict(row, train_df, k=3)
    actual = row['label']
    
    print(f"Predicted: {pred}, Actual: {actual}")
    
    if pred == actual:
        correct += 1

accuracy = correct / total
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")


y_true = []
y_pred = []

for _, row in test_df.iterrows():
    pred = knn_predict(row, train_df, k=3)
    y_pred.append(pred)
    y_true.append(row['label'])

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()