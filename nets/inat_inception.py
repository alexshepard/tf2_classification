import tensorflow as tf

# our modifications to inception
# currently just vanilla inceptionv3 with our own
# head but this will get the dual-head factorized
# stuff

def compiled_model(img_shape, image_batch, num_classes):
    base_model = tf.keras.applications.InceptionV3(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )

    feature_batch = base_model(image_batch)
    base_model.trainable = True

    # add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    prediction_batch = prediction_layer(feature_batch_average)
    
    # append the classification head to the base model
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
