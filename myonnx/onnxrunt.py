import onnxruntime
import numpy as np
import cv2 as cv

input_img = cv.imread('face.png')
input_img =cv.resize(input_img,(256,256)).astype(np.float32)
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

ort_session = onnxruntime.InferenceSession("srcnn.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv.imwrite("face_ort.png", ort_output)