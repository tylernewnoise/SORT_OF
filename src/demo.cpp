#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "sort_of.h"

using namespace sort_of;

// Got this from nwojke (https://github.com/nwojke).
namespace {
template <typename UInt>
cv::Vec3b createUniqueColorHsv(const UInt& id) {
  static constexpr double HUE_STEP{0.41};
  const double hue{std::fmod(id * HUE_STEP, 1.0)};
  const double value{1.0 - (static_cast<int>(id * HUE_STEP) % 4) / 5.0};
  static constexpr double saturation{1.0};
  return cv::Vec3b(180 * hue, 255 * value, 255 * saturation);
}
}  // namespace

// Function prototypes.
void demoSort(const std::string&, const bool&, const bool&);
std::map<std::size_t, std::vector<BBox>> readDetections(const std::string&);
std::vector<cv::Mat> readImages(const std::string&);
void visualize(const std::vector<cv::Mat>&, const std::size_t&,
               const std::vector<struct sort_of::Track>&,
               const std::vector<BBox>&);

// Ugly globals for time measurement.
std::chrono::duration<double> timespan{};
std::size_t overall_frames;

int main(int argc, char* argv[]) {
  // Data sets with static cameras only.
  std::vector<std::string> sequences = {"ADL-Rundle-6",   "KITTI-17",
                                        "PETS09-S2L1",    "TUD-Campus",
                                        "TUD-Stadtmitte", "Venice-2"};

  bool display{false};
  bool write_output{false};
  if (argc >= 2) {
    while (true) {
      int result = getopt(argc, argv, "do");
      if (result == -1) break;
      switch (result) {
        case 'd':
          display = true;
          std::cout << "Running with visualization." << std::endl;
          std::cout << "Press any key to forward and [Esc] to end."
                    << std::endl;
          break;
        case 'o':
          write_output = true;
          std::cout << "Writing output to output/SEQUENCE.txt." << std::endl;
          break;
        default:
          break;
      }
    }
  }

  if (!display)
    std::cout << "Running without visualization.\n"
              << "Use " << argv[0] << " [-d] to visualize"
              << " tracking output." << std::endl;

  std::chrono::duration<double> timespan_local{};
  auto t1 = std::chrono::high_resolution_clock::now();

  for (const auto& seq : sequences) demoSort(seq, display, write_output);

  auto t2 = std::chrono::high_resolution_clock::now();
  timespan_local.operator+=(
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1));

  std::cout << "Overall process time: " << timespan_local.count() << "\n"
            << "Total Tracking took: " << timespan.count() << " for "
            << overall_frames << " frames or "
            << ((double)overall_frames / timespan.count()) << " FPS"
            << std::endl;
  if (display)
    std::cout << "Note: get real runtime results without the option "
                 "[-d]."
              << std::endl;
  return 0;
}

/**
 * Runs SORT on a given sequence.
 *
 * @param seqName std::string with the sequence name.
 * @param display If set to true, the results will be visualized.
 * @param write_output If set to true, tracks will be saved in mot challenge
 *                     format to file.
 */
void demoSort(const std::string& seqName, const bool& display,
              const bool& write_output) {
  std::cout << "Processing " << seqName << "..." << std::endl;
  // Read detections.
  std::map<std::size_t, std::vector<BBox>> detections = readDetections(seqName);
  std::size_t max_frames = detections.rbegin()->first;

  // Read images.
  std::vector<cv::Mat> images = readImages(seqName);
  if (display) cv::namedWindow("Tracking", cv::WINDOW_AUTOSIZE);

  std::ofstream out_file;
  if (write_output) {
    // Create file for output of tracking results.
    struct stat st {};
    if (!(stat("output", &st) == 0 && S_ISDIR(st.st_mode)))
      std::filesystem::create_directory("output");

    std::string out_file_path = "output/" + seqName + ".txt";

    out_file.open(out_file_path);
    if (out_file.is_open())
      std::cout << "Tracking result in MOT format is written to "
                << out_file_path << std::endl;
    else
      std::cerr << "Error, can not write output file to " << out_file_path
                << std::endl;
  }

  // Create instance of sort tracker.
  const unsigned int max_age{1};
  const int max_corners{50};
  const unsigned int n_init{3};
  const double iou_threshold{0.3};
  sort_of::SORTOF of_sort_tracker =
      SORTOF(iou_threshold, max_age, max_corners, n_init);
  std::vector<struct sort_of::Track> tracks;

  // Iterate over frames and start tracking.
  std::cout << "Start tracking... " << std::flush;
  for (std::size_t frame = 1; frame <= max_frames; ++frame) {
#if !defined(NDEBUG)
    std::cout << " ############## Processing frame: " << frame
              << " ##############" << std::endl;
#endif
    auto it = detections.find(frame);
    if (it != detections.end()) {
#if !defined(NDEBUG)
      std::cout << "Detections to update with: " << std::endl;
      for (const auto& i : it->second) std::cout << i << ' ';
      std::cout << std::endl;
#endif
      struct sort_of::DetectionsAndImg dets_and_img;
      dets_and_img.detections = it->second;
      dets_and_img.img = images.at(frame - 1);

      auto t1 = std::chrono::high_resolution_clock::now();

      // Update the mot tracker with detections and image.
      tracks = of_sort_tracker.update(dets_and_img);

      auto t2 = std::chrono::high_resolution_clock::now();
      timespan.operator+=(
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1));

      if (display) visualize(images, frame, tracks, it->second);
    } else {
#if !defined(NDEBUG)
      std::cout << "no detections to update" << std::endl;
#endif
      auto t1 = std::chrono::high_resolution_clock::now();
      struct sort_of::DetectionsAndImg dets_and_img;
      dets_and_img.img = images.at(frame - 1);

      // Even if there are no detections we have to update the tracker.
      tracks = of_sort_tracker.update(dets_and_img);

      auto t2 = std::chrono::high_resolution_clock::now();
      timespan.operator+=(
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1));

      if (display) visualize(images, frame, tracks, std::vector<BBox>());
    }

    if (write_output) {
      // Write results.
      for (const auto& t : tracks)
        out_file << frame << "," << t.id << "," << std::fixed
                 << std::setprecision(2) << t.bbox.x << "," << t.bbox.y << ","
                 << t.bbox.width << "," << t.bbox.height << ",1,-1,-1,-1"
                 << std::endl;
    }
    ++overall_frames;
  }
  std::cout << "done." << std::endl;
}

/**
 * Reads detections in mot challenge format from file.
 *
 * @param seqName String with the processed sequence name.
 * @return detections A std::map with lists of cv::rect_ mapped to frame
 *                    numbers.
 */
std::map<std::size_t, std::vector<BBox>> readDetections(
    const std::string& seqName) {
  // Detections are represented as a map.
  std::map<std::size_t, std::vector<BBox>> detections;

  std::ifstream detectionsFileStream;
  std::string detectionsFilePath = "data/" + seqName + "/det/det.txt";
  detectionsFileStream.open(detectionsFilePath);

  if (!detectionsFileStream.is_open()) {
    std::cerr << "Error: can not open detection file " << detectionsFilePath
              << "." << std::endl;
    exit(1);
  }

  std::cout << "Reading detections into RAM... " << std::flush;

  // Read file line by line.
  std::string line;
  while (std::getline(detectionsFileStream, line)) {
    std::istringstream ss;
    ss.str(line);

    std::vector<float> detection;
    std::string data;
    // Create vector representation of line.
    while (std::getline(ss, data, ',')) detection.push_back(std::stof(data));

    // MOT challenge format should have at least 10 entries.
    if (detection.size() < 10) continue;

    // Look up the detection...
    auto frame_nr = detection[0];
    auto it = detections.find(static_cast<std::size_t>(frame_nr));
    if (it != detections.end()) {
      auto x = detection[2];
      auto y = detection[3];
      auto w = detection[4];
      auto h = detection[5];
      BBox bbox(x, y, w, h);

      it->second.push_back(bbox);
    } else {
      auto x = detection[2];
      auto y = detection[3];
      auto w = detection[4];
      auto h = detection[5];
      BBox bbox(x, y, w, h);
      std::vector<BBox> frame_detections;

      frame_detections.push_back(bbox);
      detections[static_cast<std::size_t>(frame_nr)] = frame_detections;
    }
  }

  std::cout << "done." << std::endl;
  return detections;
}

/**
 * Reads images to RAM for faster processing.
 * @param seqName String with the processed sequence name.
 * @return images List with images.
 */
std::vector<cv::Mat> readImages(const std::string& seqName) {
  std::cout << "Reading images into RAM... " << std::flush;
  std::vector<cv::String> image_files;
  std::string imagesFilePath = "data/" + seqName + "/img1";
  cv::glob(imagesFilePath, image_files);

  std::vector<cv::Mat> images;
  for (const auto& file : image_files) images.push_back(cv::imread(file));

  std::cout << "done." << std::endl;
  return images;
}

/**
 * Visualizes the tracking output if the -d flag is set.
 *
 * @param images List of images.
 * @param frame Number of frames.
 * @param tracks List of struct Track with tracks.
 * @param detections List cv::rect_ with detections.
 */
void visualize(const std::vector<cv::Mat>& images, const std::size_t& frame,
               const std::vector<struct sort_of::Track>& tracks,
               const std::vector<BBox>& detections) {
  // Read image file
  cv::Mat img = images.at(frame - 1);

  // Paint tracks.
  static constexpr int THICKNESS = 2;
  for (const auto& t : tracks) {
    cv::Mat bgr;
    cv::Mat hsv(1, 1, CV_8UC3, createUniqueColorHsv(t.id));
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Point_<float> top_left(t.bbox.x, t.bbox.y);
    const cv::Point_<float> bottom_right =
        top_left + cv::Point_<float>(t.bbox.width, t.bbox.height);

    cv::rectangle(img, top_left, bottom_right,
                  cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]), THICKNESS);

    const std::string label = std::to_string(t.id);
    int baseline;
    cv::Size_<float> text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1.0, 2.0, &baseline);

    // Paint a neat box with the tracking id.
    cv::rectangle(img, top_left,
                  cv::Point_<float>(top_left.x + 10 + text_size.width,
                                    top_left.y + 10 + text_size.height),
                  cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]), -1);

    const cv::Point_<float> text_position(top_left.x + 5,
                                          top_left.y + 5 + text_size.height);
    cv::putText(img, label, text_position, cv::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(255, 255, 255), 2.0);
  }

  // Paint detections.
  for (const auto& det : detections)
    cv::rectangle(img, det, cv::Scalar(255, 255, 255), 1);

  cv::imshow("Tracking", img);
  auto k = cv::waitKey(0);
  if (27 == k) exit(0);
}
