#! /bin/sh.

bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --verbose_failures  \
  mediapipe/examples/desktop/washoe:washoe_gpu

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/washoe/washoe_gpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt \
  --display_feedback=true \
  --camera_path="/dev/video0"


