import detr
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the coco dataset
ds, ds_info = tfds.load(
    name='coco/2017',
    # split='test[:5%]', # +train[:5%]+validation[:5%]
    data_dir='~/.tensorflow_datasets/', 
    with_info=True
)

# Optionally show examples
# tfds.show_examples(ds=ds, ds_info=ds_info)

# Function to preprocess the dataset
IMG_SIZE = 224  # Adjust size as needed

def preprocess(example):
    """Preprocess images and bounding boxes for DETR."""
    image = tf.image.resize(example['image'], (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize image
    
    # Extract bounding box and labels
    bbox = example['objects']['bbox']  # Normalized [ymin, xmin, ymax, xmax]
    label = example['objects']['label']  # Object classes

    return image, (bbox, label)

# Prepare dataset
train_ds = ds['train'].map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE) #.shuffle(1000).batch(32)
val_ds = ds['validation'].map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model = detr.DETR(num_classes=80, num_queries=100)
model.compile(optimizer='adam')
model.fit(train_ds, batch_size=32)
