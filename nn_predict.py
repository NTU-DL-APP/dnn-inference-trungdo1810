import numpy as np
import json
import os

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now

def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

from utils import mnist_reader

YOUR_MODEL_PATH = 'model/fashion_mnist' # Default format is h5
#TF_MODEL_PATH = f'{YOUR_MODEL_PATH}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_PATH}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_PATH}.json'
OUTPUT_FILE = 'test_acc.txt'

def test_inference():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    acc = None
    try:
        x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

        # === Load weights and architecture ===
        weights = np.load(MODEL_WEIGHTS_PATH)
        with open(MODEL_ARCH_PATH) as f:
            model_arch = json.load(f)
            
        # Shuffle the test dataset
        indices = np.arange(x_test.shape[0])
        np.random.shuffle(indices)
        x_test_shuffled = x_test[indices]
        y_test_shuffled = y_test[indices]
        
        # Normalize input images
        normalized_X = x_test_shuffled / 255.0
        
        # Perform inference for all test images
        print('Classifying images...')
        outputs = np.array([
            nn_inference(model_arch, weights, np.expand_dims(img, axis=0))
            for img in normalized_X
        ])
        print('Done')

        # Get predictions using argmax
        predictions = np.argmax(outputs.squeeze(axis=1), axis=-1)

        # Calculate number of correct predictions
        correct = np.sum(predictions == y_test_shuffled)

        acc = correct / len(y_test_shuffled)
        print(f"Accuracy = {acc}")
        with open(OUTPUT_FILE, 'w') as file:
            file.write(str(acc))

    except Exception as e:
        print("Error! ", e)

    assert acc != None

if __name__ == "__main__":
    # weights = np.load(MODEL_WEIGHTS_PATH)
    # print("Keys in .npz file:", weights.files)

    # with open(MODEL_ARCH_PATH) as f:
    #     model_arch = json.load(f)
    # print("Model architecture:", model_arch)
    test_inference()
    print("Test inference completed successfully.")