import keras
import matplotlib.pyplot as plt

# Load the model without the fully connected layers
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.summary()

# Choose an intermediate layer to extract features
layer_name = 'conv3_block4_out'  # Example layer
model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# Load an image and resize to 224x224 pixels (ResNet50 input size)
img_path = 'test.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

# Convert image to an array and expand dimensions for batch processing
img_array = keras.preprocessing.image.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, axis=0)

# Preprocess the image for ResNet50
img_array = keras.applications.resnet50.preprocess_input(img_array)

# Predict feature maps from the selected layer
feature_maps = model.predict(img_array)

# Display feature map shape
print("Feature map shape:", feature_maps.shape)  # Example: (1, 14, 14, 1024)

# Visualize first 16 feature maps
num_feature_maps = min(16, feature_maps.shape[-1])  # Show up to 16 feature maps
plt.figure(figsize=(7, 7))

for i in range(num_feature_maps):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')

plt.show()