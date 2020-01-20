import tensorflow as tf

# our modifications to inception
# currently just vanilla inceptionv3 with our own
# head but this will get the dual-head factorized
# stuff

def compiled_model(img_shape, num_classes):
    base_model = tf.keras.applications.InceptionV3(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = True

    # add a classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name='pool1')(base_model.output)
    pred1 = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='head'
    )(x)


    model = tf.keras.models.Model(
        base_model.input,
        pred1
    )

    base_learning_rate = 0.0001

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=base_learning_rate
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
