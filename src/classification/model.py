import tensorflow as tf


def load_split(classification_dir, split, img_size, batch_size, seed, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        classification_dir / split,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def build_model(num_classes, img_size, dropout):
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="inceptionv3_classifier"), base_model


def unfreeze_from(model, base_model, block_name):
    base_model.trainable = True
    trainable = False
    for layer in base_model.layers:
        if layer.name == block_name:
            trainable = True
        layer.trainable = trainable and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return sum(int(tf.size(w)) for w in model.trainable_weights)
