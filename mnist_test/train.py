import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import StandardScaler
import numpy as np

mnist = input_data.read_data_sets("./data")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

feature_columns = [tf.feature_column.numeric_column("x", shape=[X_train.shape[1]])]


# classifier = tf.estimator.DNNClassifier(hidden_units=[300,100],
#                                         n_classes=10,
#                                         feature_columns=feature_columns,
#                                         model_dir="model",
#                                         optimizer=tf.train.ProximalAdagradOptimizer(
#                                         learning_rate=0.1,
#                                         l1_regularization_strength=0.001))

classifier = tf.estimator.DNNClassifier(hidden_units=[300,100],
                                        n_classes=10,
                                        feature_columns=feature_columns,
                                        model_dir="model",
                                        dropout=0.01,
                                        activation_fn=tf.nn.elu)



train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": X_train},
  y=np.array(y_train),
  num_epochs=1000,
  shuffle=True)

classifier.train(input_fn=train_input_fn, steps=40000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_test)},
    y=np.array(y_test),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
class_eval = classifier.evaluate(input_fn=test_input_fn)
#accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print(class_eval)