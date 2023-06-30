
#Implementation example of Grover Operator
import numpy as np

def grover_operator(amplitudes):
    # Apply inversion about the average
    average = np.mean(amplitudes)
    amplitudes = (2 * average) - amplitudes

    # Apply phase inversion
    amplitudes *= -1

    return amplitudes

# Example usage
# Suppose we have a list with 8 elements and we want to find the marked item at index 3

# Initialize the amplitudes with equal superposition
n = 8  # Number of elements in the list
amplitudes = np.full(n, 1/np.sqrt(n))

# Mark the item at index 3 by negating its amplitude
marked_index = 3
amplitudes[marked_index] *= -1

# Apply the Grover operator
amplitudes = grover_operator(amplitudes)

# Print the resulting amplitudes
for i, amplitude in enumerate(amplitudes):
    print(f"Amplitude of index {i}: {amplitude}")


########################### Example of custom matrix creation for quantum operator
import numpy as np

# Define the custom transformation
transformation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

# Check if the transformation matrix is unitary
is_unitary = np.allclose(np.conj(transformation.T) @ transformation, np.identity(2))

# Print the transformation matrix and unitarity check
print("Custom Transformation Matrix:")
print(transformation)
print("\nIs Unitary:", is_unitary)
