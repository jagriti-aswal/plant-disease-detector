import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = tf.keras.models.load_model(model_path)

# Last conv layer (MobileNetV2)
last_conv_layer_name = "Conv_1"


def get_gradcam(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Preprocess
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Create grad model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weighted sum
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize safely
    heatmap = tf.maximum(heatmap, 0)
    if tf.math.reduce_max(heatmap) != 0:
        heatmap /= tf.math.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert to color map
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    final_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return final_img


# Test locally
if __name__ == "__main__":
    result = get_gradcam("test.jpg")
    cv2.imwrite("gradcam_output.jpg", result)