import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	# output matrix size:
	output_height = (input_height - kernel_height + 2 * padding) // stride + 1
	output_width = (input_width - kernel_width + 2 * padding) // stride + 1

	padded_matrix = np.pad(input_matrix, padding, "constant")

	output_matrix = np.zeros((output_height, output_width))

	for y in range(output_height):
		for x in range(output_width):
			row_start = y * stride
			row_end = row_start + kernel_height
			col_start = x * stride
			col_end = col_start + kernel_width

			window = padded_matrix[row_start:row_end, col_start:col_end]

			output_matrix[y, x] = np.sum(window * kernel)

	return output_matrix

    
	
