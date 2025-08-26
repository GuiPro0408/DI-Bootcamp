import numpy as np

temperatures = np.array([20, 22, 25, 28, 30, 32, 35])
random_values = np.random.randint(1, 11, size=(2, 5))  # 1 - 10 is due to the range of valid values for random integers
picture_pixels = np.random.randint(0, 256, size=(2, 3, 4))  # 0 - 255 is due to the range of valid pixel values

print(f"Temperatures: \n{temperatures}\nShape: {temperatures.shape}\nSize: {temperatures.size}\nData type: {temperatures.dtype}\n\n")
print(f"Random Values: \n{random_values}\nShape: {random_values.shape}\nSize: {random_values.size}\nData type: {random_values.dtype}\n\n")
print(f"Picture Pixels: \n{picture_pixels}\nShape: {picture_pixels.shape}\nSize: {picture_pixels.size}\nData type: {picture_pixels.dtype}\n\n")
