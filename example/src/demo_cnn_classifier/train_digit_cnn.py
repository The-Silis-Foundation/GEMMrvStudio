import numpy as np
import cv2
import os

def im2col1(img):
    cols = []
    for r in range(0, 24, 2):
        for c in range(0, 24, 2):
            cols.append(img[r:r+5, c:c+5].flatten())
    return np.array(cols)

def im2col2(feat):
    cols = []
    for r in range(0, 9, 2):
        for c in range(0, 9, 2):
            cols.append(feat[r:r+3, c:c+3, :].flatten())
    return np.array(cols)

# 10 Digit Dataset
X_list, y_list = [], []
for d in range(10):
    char = str(d)
    for _ in range(100):
        img = np.zeros((48, 48), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = np.random.uniform(1.0, 1.25)
        thick = np.random.choice([1, 2])
        (tw, th), _ = cv2.getTextSize(char, font, scale, thick)
        x = (48 - tw) // 2 + np.random.randint(-2, 3)
        y = (48 + th) // 2 + np.random.randint(-2, 3)
        cv2.putText(img, char, (x, y), font, scale, 255, thick, cv2.LINE_AA)
        img28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        norm = (img28.astype(np.float32) / 255.0)
        X_list.append(norm)
        y_list.append(d)

X_arr = np.array(X_list)
y_arr = np.array(y_list)

np.random.seed(42)
W_c1 = np.random.randn(25, 8) * 0.1
b_c1 = np.zeros(8)
W_c2 = np.random.randn(72, 16) * 0.05
b_c2 = np.zeros(16)
W_fc = np.random.randn(400, 40) * 0.05
b_fc = np.zeros(40)

lr = 0.03
for epoch in range(12):
    perm = np.random.permutation(len(X_arr))
    X_s = X_arr[perm]
    y_s = y_arr[perm]
    for i in range(len(X_s)):
        col1 = im2col1(X_s[i])
        c1 = np.maximum(0, col1 @ W_c1 + b_c1)
        feat1 = c1.reshape(12, 12, 8)
        col2 = im2col2(feat1)
        c2 = np.maximum(0, col2 @ W_c2 + b_c2)
        feat2 = c2.flatten()
        logits = feat2 @ W_fc + b_fc
        
        ex = np.exp(logits[:10] - np.max(logits[:10]))
        p = ex / np.sum(ex)
        target = y_s[i]
        
        dl = np.zeros(40)
        dl[:10] = p
        dl[target] -= 1.0
        
        g_W_fc = np.outer(feat2, dl)
        g_b_fc = dl
        dfeat2 = dl @ W_fc.T
        dc2 = dfeat2.reshape(25, 16) * (c2 > 0)
        g_W_c2 = col2.T @ dc2
        g_b_c2 = np.sum(dc2, axis=0)
        
        dcol2 = dc2 @ W_c2.T
        dfeat1 = np.zeros((12, 12, 8))
        idx = 0
        for r in range(0, 9, 2):
            for c in range(0, 9, 2):
                dfeat1[r:r+3, c:c+3, :] += dcol2[idx].reshape(3, 3, 8)
                idx += 1
        dc1 = dfeat1.reshape(144, 8) * (c1 > 0)
        g_W_c1 = col1.T @ dc1
        g_b_c1 = np.sum(dc1, axis=0)
        
        W_fc -= lr * g_W_fc
        b_fc -= lr * g_b_fc
        W_c2 -= lr * g_W_c2
        b_c2 -= lr * g_b_c2
        W_c1 -= lr * g_W_c1
        b_c1 -= lr * g_b_c1

# Quantize to INT8:
scale = 64.0
def quant(mat):
    return np.clip(np.round(mat * scale), -127, 127).astype(np.int8)

W_c1_q = quant(W_c1)
W_c2_q = quant(W_c2)
W_fc_q = quant(W_fc)

b_c1_q = np.round(b_c1 * scale * scale).astype(np.int32)
b_c2_q = np.round(b_c2 * scale * scale).astype(np.int32)
b_fc_q = np.round(b_fc * scale * scale).astype(np.int32)

W_c1_pad = np.zeros((32, 8), dtype=np.int8)
W_c1_pad[:25, :8] = W_c1_q

targets = [
    r"C:\Users\Lenovo\Desktop\scripts\demo_cnn_classifier\cnn_model_weights.h",
    r"C:\Users\Lenovo\Desktop\GEMMrv32\GEMMrv32-main\sw\example\demo_cnn_classifier\cnn_model_weights.h"
]

for out_h in targets:
    with open(out_h, "w", encoding="utf-8") as out_f:
        out_f.write("#ifndef CNN_MODEL_WEIGHTS_H\n#define CNN_MODEL_WEIGHTS_H\n\n#include <stdint.h>\n\n")
        out_f.write("#define CNN_NUM_CLASSES 10\n")
        out_f.write("#define CNN_OUT_DIM     40\n\n")
        out_f.write('static const char CNN_CLASSES[11] = "0123456789";\n\n')
        
        def dump_arr(arr, name, dtype="int8_t"):
            out_f.write(f"static const {dtype} {name}[{arr.size}] __attribute__((aligned(32))) = {{\n")
            flat = arr.flatten()
            stride = 16 if dtype == "int8_t" else 8
            for k in range(0, len(flat), stride):
                out_f.write("    " + ", ".join(str(int(val)) for val in flat[k:k+stride]) + ",\n")
            out_f.write("};\n\n")
            
        dump_arr(W_c1_pad, "W_CONV1")
        dump_arr(b_c1_q, "B_CONV1", "int32_t")
        dump_arr(W_c2_q, "W_CONV2")
        dump_arr(b_c2_q, "B_CONV2", "int32_t")
        dump_arr(W_fc_q, "W_FC")
        dump_arr(b_fc_q, "B_FC", "int32_t")
        out_f.write("#endif // CNN_MODEL_WEIGHTS_H\n")

print("Successfully written 10-digit weights to cnn_model_weights.h!")
