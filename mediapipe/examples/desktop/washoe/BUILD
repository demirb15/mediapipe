# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "washoe_gpu",
    deps = [
        ":washoe_desktop_live_gpu",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
    ],
)

# Linux only
cc_library(
    name = "washoe_desktop_live_gpu",
    srcs = ["target.cc"],
    visibility = ["//visibility:public"],
    hdrs = glob([
        "include/zmq.hpp",
        "include/zmq_addon.hpp",
    ]),
    deps = [
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
    linkopts = [
        "-lzmq ",
    ]
)
