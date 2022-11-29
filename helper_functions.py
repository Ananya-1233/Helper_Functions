def loss_and_accuracy(history):
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs , loss , label = 'training_loss')
  plt.plot(epochs , val_loss , label = 'validation_loss')
  plt.title('Loss Curve')
  plt.xlabel('epochs')
  plt.legend()
  
  plt.figure()

  plt.plot(epochs , accuracy , label = 'training_accuracy')
  plt.plot(epochs , val_accuracy , label = 'validation_accuracy')
  plt.title('Accuracy Curve')
  plt.xlabel('epochs')
  plt.legend()
    
    
def load_image(filename , img_shape=224):

  img = tf.io.read_file(filename)

  img = tf.image.decode_image(img , channels = 3)

  img = tf.image.resize(img , size = [img_shape , img_shape])

  img = img/255.

  return img


def make_predictions(model , filename , classnames):

  image = load_image(filename)

  pred = model.predict(tf.expand_dims(image , axis=0))

  pred_class = classnames[tf.argmax(tf.round(pred)[0])]

  plt.imshow(image)
  plt.title(f'Predicted class: {pred_class}')

  plt.axis(False)
