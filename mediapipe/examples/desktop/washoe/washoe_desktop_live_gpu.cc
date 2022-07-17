// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.

#include <string>
#include <zmq_addon.hpp>
#include <zmq.hpp>
#include <ctime>
#include <sstream>

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kDetectionsStream[] = "hand_landmarks";
constexpr char kLandmarkPresenceStream[] = "landmark_presence";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(bool, display_feedback, false,
          "Boolean value to display current video capture");
ABSL_FLAG(std::string, camera_path, "",
          "Full path to camera device"
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, socket_address, "tcp://127.0.0.5:42207",
          "Port number of socket"
          "If not provided, defaults to tcp://127.0.0.5:42207");


absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera";
  cv::VideoCapture capture;
  const bool use_camera_path = !absl::GetFlag(FLAGS_camera_path).empty();
  if (use_camera_path) {
    capture.open(absl::GetFlag(FLAGS_camera_path));
  } else {
    capture.open(0);
  }
  const bool should_display_feedback = absl::GetFlag(FLAGS_display_feedback);
  RET_CHECK(capture.isOpened());

  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 60);
#endif
  LOG(INFO) << "Start running the calculator graph.";
  // ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller video_poller,
  //                  graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller,
                   graph.AddOutputStreamPoller(kDetectionsStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller presence_poller,
                   graph.AddOutputStreamPoller(kLandmarkPresenceStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  int turn = 0;
  // ------------------------------------------
  zmq::context_t ctx;
  zmq::socket_t publisher(ctx, zmq::socket_type::push);
  std::string address = absl::GetFlag(FLAGS_socket_address);
  publisher.bind(address);
  const std::string last_endpoint = publisher.get(zmq::sockopt::last_endpoint);
  LOG(INFO) << "Connected to " << last_endpoint;
  // ------------------------------------------
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!use_camera_path) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGBA);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return absl::OkStatus();
        }));

    // Convert back to opencv for display
    if (should_display_feedback) {
      cv::Mat output_frame_mat =
          mediapipe::formats::MatView(input_frame.get());
      if (output_frame_mat.channels() == 4)
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
      else
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) {
        grab_frames = false;
        continue;
      }
      // Display window
      cv::imshow(kWindowName, output_frame_mat);
    }
    //================== Send landmarks ===================
    turn++;
    switch (turn % 4)
    {
      case 0:
        std::cout << "\rrunning /" << std::flush;
        break;
      case 1:
        std::cout << "\rrunning |" << std::flush;
        break;
      case 2:
        std::cout << "\rrunning \\" << std::flush;
        break;
      case 3:
        std::cout << "\rrunning —" << std::flush;
        break;
      default:
        std::cout << "\rrunning 0" << std::flush;
        break;
    }
              
    // Check if hand landmarks is present
    mediapipe::Packet presence_packet;
    if (!presence_poller.Next(&presence_packet)) break;
    auto is_landmark_present = presence_packet.Get<bool>();
    if (!is_landmark_present) continue;

    // Broadcast landmarks
    mediapipe::Packet landmark_packet;
    if (!landmark_poller.Next(&landmark_packet))
      break;

    auto output_landmarks =
        landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    // std::cout << "Number of hands :: " << output_landmarks.size() << std::endl;
    for (int i = 0; i < output_landmarks.size(); ++i) {
      // std::cout << output_landmarks[i].DebugString() << std::endl;
      auto landmarks = output_landmarks[i].landmark();
      // std::cout << '\r' << landmarks[0].DebugString() << std::flush << std::endl;
      std::ostringstream oss;
      oss <<  landmarks[0].DebugString();
      std::string str = oss.str();
      const char* cstr = str.c_str();
      char buffer[49];
      strcpy(buffer, str.c_str());
      publisher.send(zmq::str_buffer(buffer), zmq::send_flags::dontwait);
    }
  }

  LOG(INFO) << "Shutting down.";
  publisher.close();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
