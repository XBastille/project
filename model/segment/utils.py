import numpy as np
import math

def calculate_padding(image, patch_size=256):
  """
    Calculates required padding amounts
    ARGS:
      image: Image to be padded (height, width, channels)
      patch_size: Stride of the patches
    RETURNS:
      padding: Padding amounts (top, bottom, left, right)
  """

  height, width = image.shape[-3:-1]

  # Calculate Desired shape
  xx = (width // patch_size +1) * patch_size
  yy = (height // patch_size +1) * patch_size

  # Calculate padding amounts
  x_pad = xx - width
  y_pad = yy - height

  x0 = x_pad // 2
  x1 = x_pad - x0
  y0 = y_pad // 2
  y1 = y_pad - y0

  return (y0, y1, x0, x1)

def apply_padding(image, patch_size=256):
  """
    Adds padding to the image to make patches of equal size that covers the entire image
    ARGS:
      image: Image to be padded
      patch_size: Stride of the patches
    RETURNS:
      padded_image: Padded image
  """
  width, height = image.shape[:2]
  pad_y0, pad_y1, pad_x0, pad_x1 = calculate_padding(image, patch_size)

  return np.pad(image, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode='constant')


def predict(model, img, threshold=0.5, patch_size=256):
  """
    model: a keras model
    img: a np array representing the image (expects float dtype and values between 0 and 1)
  """
  padded = apply_padding(img, patch_size)

  number_of_rows, number_of_columns = math.ceil(img.shape[0] /patch_size), math.ceil(img.shape[1] /patch_size)
  # fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(20, 20))
  prediction = np.zeros((number_of_rows * patch_size, number_of_columns * patch_size, 1), dtype="float32")

  for i in range(0, number_of_rows):
    for j in range(0, number_of_columns):
      # Convert to float
      slc = padded[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] / 256.0
      pred = model.predict(np.expand_dims(slc, axis = 0), verbose = 0)
      # axes[i][j].imshow(pred[0])
      prediction[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred[0]

  # Threshold the results
  prediction = (prediction > threshold).astype(np.uint8)

  # Reverse padding
  y0, y1, x0, x1 = calculate_padding(img, patch_size)
  preds = prediction[y0:-y1, x0:-x1]
  return preds