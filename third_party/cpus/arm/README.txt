To cross compile TFLITE use the follow command line
bazel build --crosstool_top=third_party/toolchains/cpus/arm:toolchain --cpu=armeabi tensorflow/contrib/lite/examples/simplelite:simplelite
