import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite
interpreter = tflite.Interpreter(model_path='efficientdet_lite0_fp16_2.tflite')
interpreter.allocate_tensors()

print('Model loaded successfully!')
