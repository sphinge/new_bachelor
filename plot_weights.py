import matplotlib.pyplot as plt
import csv

# Read data from the CSV file
layer0_weights = []
layer1_weights = []
loss_values = []

with open('weights.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row[0].startswith('Layer 0:'):
            layer0_weights.append([float(val) for val in row[1:]])
        elif row[0].startswith('Layer 1:'):
            layer1_weights.append([float(val) for val in row[1:]])
        elif row[0].startswith('i:'):
            loss = float(row[-1].split()[-1])
            loss_values.append(loss)

# Create plots without iteration labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for values in layer0_weights:
    plt.plot(values)
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Hidden Layer Weights')

plt.subplot(1, 2, 2)
for values in layer1_weights:
    plt.plot(values)
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('Output Layer Weights')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(loss_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss over Iterations')
plt.grid()
plt.show()
