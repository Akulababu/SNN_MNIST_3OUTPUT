I. Main Code:
Indataprocessing.py is the data preprocessing code, used to convert data into a form that neural networks can process.
gennet.py is the neural network definition code, including the network structure and parameters.
Trainloop8.py contains the main steps of the training loop, including training, testing, and evaluation.
maintrain.py is the entry point for the training loop, used to start the training process.

II. Weight Files:
final_weights.txt is the weight file of the trained neural network, including network structure and parameters.

III. Inference Code:
inference.py is the inference code, used to apply the trained neural network to new data.
correct_case.txt is a file containing correctly predicted cases after running the inference code, including input data and prediction results.

IV. Pure Mathematical Model Verification Code:
verify.py is the verification code for the pure mathematical model, involving only matrix operations, used to verify neural network performance.
verification.txt is the verification results file for the pure mathematical model, including accuracy, loss values, etc.
verification_differences.txt shows the differences between the pure mathematical model and inference model on the validation set.

The remaining Python files are hardware-related code, including hardware implementation and verification of spiking neural networks.

V. Structural Overview
My neural network first converts 2x3x4 data from 28x28 images to 7x7 images before training.
The first layer has 49 neurons, the hidden layer has 24, and the output layer has 3.
The decay value is 0, GPU is used for batch sample training, and neurons are reset before each batch training. Neurons have no memory function.
Thresholds are 1, 1.99, 1.99, because SNNtorch's library code judges threshold firing principle as greater than, while we want greater than or equal to. I found that modifying the library code causes training issues, so it's set to 1.99.
The loss consists of two parts: spike loss and membrane loss.
Spike loss is cross-entropy loss, membrane loss is also cross-entropy loss.
For weight restrictions, I used weight clipping to limit weights between -1 and +1, maintaining normalized weights, while strictly limiting each neuron to a maximum of 6 positive weight connections and 2 negative weight connections.
Pruning of the hidden layer begins at 30 epochs, limiting each neuron's total connections to 8, and pruning of the output layer begins at 50 epochs, limiting each neuron's total connections to 8.
Quantization of the hidden layer to three values (-1, 0, +1) begins at 70 epochs, and input layer quantization begins 30 epochs later.

Current accuracy is 79.6%


