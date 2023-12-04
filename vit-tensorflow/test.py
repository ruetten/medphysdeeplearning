import tensorflow as tf
from vit_tensorflow.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

v.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

img = tf.random.normal([1, 256, 256, 3])
labels = tf.constant([0])

v.fit(x=img, y=labels, epochs=10)

preds = v(img) # (1, 1000)

print(preds)
