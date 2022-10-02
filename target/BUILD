package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "target",
    deps = [
        ":target_cc",
    ],
)

cc_library(
    name = "target_cc",
    srcs = ["target.cc"],
    visibility = ["//visibility:public"],
    hdrs = glob([
    "include/zmqq.hpp",
    "include/zmqq_addon.hpp"
    ]),
    deps = [
        "@mediapipe//mediapipe/graphs/hand_tracking:mobile_calculators",
        "@mediapipe//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_presence_calculator",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/gpu:gl_calculator_helper",
        "@mediapipe//mediapipe/gpu:gpu_buffer",
        "@mediapipe//mediapipe/gpu:gpu_shared_data_internal",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
    linkopts = [
        "-lzmq ",
    ]
)
