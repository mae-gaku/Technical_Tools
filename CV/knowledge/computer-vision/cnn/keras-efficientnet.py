import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import seaborn as sns


IMAGE_SHAPE = (224, 224)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,  validation_split = 0.2)
train_data = image_generator.flow_from_directory("", target_size=IMAGE_SHAPE, subset = "training" )
val_data = image_generator.flow_from_directory("", target_size=IMAGE_SHAPE, subset = "validation" )


for train_batch, train_label_batch in train_data:
  print("train_Image batch shape: ", train_batch.shape)
  print("train_Label batch shape: ", train_label_batch.shape)
  break
for val_batch, val_label_batch in val_data:
  print("val_Image batch shape: ", val_batch.shape)
  print("val_Label batch shape: ", val_label_batch.shape)
  break
  
feature_extractor_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

feature_extractor_layer.trainable = False

from efficientnet.tfkeras import EfficientNetB0


eff = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3))
x = tf.keras.layers.GlobalAveragePooling2D()(eff.output)
output = tf.keras.layers.Dense(2, activation='softmax', name='last_output')(x)
model = tf.keras.Model(inputs=eff.inputs, outputs=output, name='model')

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(lr=0.01),
  loss='categorical_crossentropy',
  metrics=['acc'])
  
  
  
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
 
  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()
    

epochs = 10

train_steps_per_epoch = np.ceil(train_data.samples/train_data.batch_size)
val_steps_per_epoch = np.ceil(val_data.samples/val_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(train_data, epochs=epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_data,
                    validation_steps=val_steps_per_epoch)    
                    
sns.set()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.savefig("sample.jpg")