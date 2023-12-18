# This import under here is like a debugger
import numpy as np

arrays = np.array([[3, 3, 3, 3, 3],
                   [2, 2, 2, 2, 2],
                   [1, 1, 1, 1, 1]])

# Calculate the federated average
average_result = np.mean(arrays, axis=0)

print(average_result)