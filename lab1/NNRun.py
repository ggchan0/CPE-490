import numpy as np
import sys

nn = sys.argv[1]
lines = []
layers = []
line_no = 1

with open(nn, 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.isspace() == False]

(numInputs, numLayers) = lines[0].split()

for i in range(0, int(numLayers)):
    num_weights = int(lines[line_no])
    line_no += 1
    weights = [l.split() for l in lines[line_no: line_no + num_weights]]
    layers.append(np.array(weights).astype(float))
    line_no += num_weights

while True:
    nums = input().split()
    layer_input = np.array(nums + [1]).astype(int).transpose()
    temp_result = layer_input
    for layer in layers:
        temp_result = np.heaviside(np.dot(layer, temp_result), 1)
        temp_result = np.append(temp_result, [1], axis = 0)
    print(temp_result)
    print(layer_input)
