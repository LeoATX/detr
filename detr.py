import keras
import keras_hub


class DETR(keras.Model):
    """
    A simple high level implementation of the DETR model with Keras and Keras Hub (previously Keras NLP).

    ```python
    model = Detr()
    bbox_preds, class_preds = model(image)
    ```
    """

    def __init__(self, num_classes: int = 100, num_queries: int = 100):
        super().__init__()
        self.backbone = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=None,
            classes=1000,
            classifier_activation='softmax',
            name='resnet50',
        )

        backbone_output_layer = 'conv4_block6_out'
        self.backbone = keras.Model(
            inputs=self.backbone.input,
            outputs=self.backbone.get_layer(backbone_output_layer).output
        )

        self.pos_encoder = keras_hub.layers.SinePositionEncoding()
        self.proj = keras.layers.Conv2D(filters=256, kernel_size=1)

        # transformer layers
        self.encoders = [
            keras_hub.layers.TransformerEncoder(intermediate_dim=2048, num_heads=6) for _ in range(8)
        ]
        self.decoders = [
            keras_hub.layers.TransformerDecoder(intermediate_dim=2048, num_heads=6) for _ in range(8)
        ]

        self.query_embed = self.add_weight(
            shape=(num_queries, 256),
            initializer='random_normal',
            trainable=True
        )

        self.class_embed = keras.layers.Dense(
            units=num_classes + 1, activation='softmax')
        # simple MLP
        self.bbox_embed = [
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=4, activation='sigmoid')
        ]

    def call(self, x):
        print('Batch size of x', keras.ops.shape(x)[0])
        x = self.backbone(x)
        x = self.proj(x)

        BATCH_SIZE, H, W, C = keras.ops.shape(x)
        # BATCH_SIZE = 32
        print('Shape of x after proj', keras.ops.shape(x))
        # flatten
        x = keras.ops.reshape(x, (BATCH_SIZE, H * W, C))

        pos_encoding = self.pos_encoder(x)
        x = x + pos_encoding

        for encoder in self.encoders:
            x = encoder(x)

        expanded_query_embed = keras.ops.expand_dims(self.query_embed, axis=0)
        expanded_query_embed = keras.ops.tile(
            expanded_query_embed, [BATCH_SIZE, 1, 1])

        print('Shape of inputs before decoder:', keras.ops.shape(x))
        for decoder in self.decoders:
            x = decoder(expanded_query_embed, x)

        class_output = self.class_embed(x)
        bbox_output = x
        for layer in self.bbox_embed:
            bbox_output = layer(bbox_output)

        return bbox_output, class_output
    
    def train_step(self, data):
        images, targets = data
        # targets might be a dict: {"boxes": ..., "labels": ...}
        
        with tf.GradientTape() as tape:
            # 1) Forward pass
            pred_boxes, pred_class_logits = self(images, training=True)
            
            # 2) Hungarian (or bipartite) matching: match each gt box to a predicted query
            matched_indices = hungarian_matching(
                pred_boxes, pred_class_logits, targets["boxes"], targets["labels"]
            )
            
            # 3) Once matched, compute classification loss and bounding box loss
            # E.g.:
            loss_cls = classification_loss_function(
                pred_class_logits, targets["labels"], matched_indices
            )
            loss_box = bbox_regression_loss_function(
                pred_boxes, targets["boxes"], matched_indices
            )
            total_loss = loss_cls + loss_box
        
        # 4) Gradients and optimization
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": total_loss, "cls_loss": loss_cls, "box_loss": loss_box}


if __name__ == '__main__':
    model = DETR()
    image = keras.utils.load_img(
        path='assets/test.jpg',
        color_mode='rgb',
        target_size=None,
        interpolation='nearest',
        keep_aspect_ratio=False,
    )
    image = keras.utils.img_to_array(image)
    image = keras.layers.Resizing(224, 224)(image)
    image = keras.ops.reshape(x=image, newshape=(1, *image.shape))
    model.summary()
    bbox_preds, class_preds = model(image)
    print(bbox_preds)
    print(class_preds)
