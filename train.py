import detr
import keras
import tensorflow_datasets as tfds

model = detr.DETR(num_classes=80, num_queries=100)
ds, ds_info = tfds.load(
    name='coco/2017',
    split='test[:5%]', # +train[:5%]+validation[:5%]
    data_dir='~/.tensorflow_datasets/', 
    with_info=True
)

print(ds_info)