import tensorflow as tf

# our modifications to inception
# currently just vanilla inceptionv3 with our own
# head but this will get the dual-head factorized
# stuff

def compiled_model(img_shape, num_classes):
    base_model = tf.keras.applications.NASNetMobile(
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
        name='head1'
    )(x)

    # add a second classification head
    # this is functionally identical to head1 except we add dropout
    # for now, anyways
    x2 = tf.keras.layers.GlobalAveragePooling2D(name='pool2')(base_model.output)
    x2 = tf.keras.layers.Dropout(0.2)(x2)
    pred2 = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='head2'
    )(x2)

    model = tf.keras.models.Model(
        base_model.input,
        [pred1, pred2]
    )

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss=['categorical_crossentropy','categorical_crossentropy'],
        metrics=['accuracy']
    )

    return model
