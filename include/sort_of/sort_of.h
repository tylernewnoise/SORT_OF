/**
 * SORT_OF: A Simple, Online and Realtime Tracker extended with Optical Flow.
 * This is base on the 2016 by Alex Bewley proposed tracking algorithm.
 * See https://github.com/abewley/sort and
 * @inproceedings{sort2016,
 * author = {Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
 * booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
 * title = {Simple online and realtime tracking},
 * year = {2016},
 * pages = {3464-3468}
 * }
 * for further information.
 *
 * This was written by Falko Becker, tyler.newnoise@gmail.com, 2021.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef SORT_OF_SORT_OF_H
#define SORT_OF_SORT_OF_H

#include <algorithm>
#include <map>
#include <vector>

#include <dlib/optimization/max_cost_assignment.h>
#include <opencv2/optflow/rlofflow.hpp>
#include <opencv2/video/tracking.hpp>

#if !defined(NDEBUG)
#include <string>
#endif

#define BBox cv::Rect_<float>

struct DetectionsAndImg {
  std::vector<BBox> detections;
  cv::Mat img;
};

struct Track {
  BBox bbox;
  std::size_t id{};
};

namespace sort_of {
/**
 * This class implements a simple multi target tracker. It holds an
 * unordered map with the track ids as the key and a KalmanBoxTracker object as
 * the value.
 */
class SORTOF {
public:
  /**
   * Constructor.
   *
   * @param iou_threshold Minimum intersection over union threshold.
   * @param max_age Maximum number of missed misses before track is deleted.
   * @param max_corners Maximum number of feature points to be detected per
   *                    track.
   * @param n_init Number of consecutive detections before the track is
   *        confirmed.
   */
  SORTOF(const double iou_threshold,
      const unsigned int max_age,
      const int max_corners,
          const unsigned int n_init)
          :
          frame_{0},
          id_{0},
          max_age_{max_age},
          n_init_{n_init},
          precision_{10000},
          trackers_{std::unordered_map<std::size_t,
                    std::unique_ptr<FlowBoxTracker>>()}
  {
    iou_threshold_ = iou_threshold * precision_;

    fast_ = cv::FastFeatureDetector::create();
    fast_->setThreshold(50);
    fast_->setNonmaxSuppression(true);
    fast_->setType(cv::FastFeatureDetector::TYPE_7_12);

    orb_ = cv::ORB::create(max_corners);
    orb_->setScaleFactor(1.2);
    orb_->setEdgeThreshold(9);
    orb_->setNLevels(8);
    orb_->setFirstLevel(0);
    orb_->setScoreType(cv::ORB::HARRIS_SCORE);
    orb_->setWTA_K(1);

    rlof_params_ = cv::optflow::RLOFOpticalFlowParameter::create();
    rlof_params_->useIlluminationModel = false;
    rlof_params_->useInitialFlow = false;
    rlof_params_->maxIteration = 5;
    rlof_params_->setUseMEstimator(false);
    rlof_params_->setUseGlobalMotionPrior(true);
    rlof_params_->setSupportRegionType(cv::optflow::SR_FIXED);
  }

  ~SORTOF() = default;

  /**
  * Perform measurement update and track management.
  *
  * @param dets_and_img A struct DetectionsAndImg with a list of detections and
  *                     a corresponding image at the current time step.
  * @return The list with active tracks at the current time step.
  */
  [[nodiscard]] std::vector<struct Track> update(
      const struct DetectionsAndImg& dets_and_img)
  {
    ++frame_;
    // Init helper variables on first frame.
    if (frame_ == 1) {
      img_height_ = float(dets_and_img.img.rows);
      img_width_ = float(dets_and_img.img.cols);
      image_space_ = BBox(0, 0, img_width_, img_height_);
    }

    // Get predicted locations from existing trackers.
    for (auto& track : trackers_)
      track.second->predict();

    // Associate detections to active trackers.
    std::vector<std::size_t> unmatched_dets;
    associate_detections_to_trackers(dets_and_img.detections, unmatched_dets);

    // Collect inactive trackers.
    std::vector<std::size_t> tracks_to_delete;
    for (auto& tracker : trackers_) {
            if (((tracker.second->time_since_update > max_age_) &&
            !tracker.second->got_bbox) ||
            (tracker.second->time_since_update == 1 &&
            tracker.second->hits < n_init_ && !tracker.second->got_bbox)) {
              tracks_to_delete.emplace_back(tracker.first);
            }
    }
    // Delete inactive trackers.
    for (const auto& trK_to_del : tracks_to_delete)
      trackers_.erase(trK_to_del);

    tracks_to_delete.clear();

    cv::Mat frame_gray;
    cv::cvtColor(dets_and_img.img, frame_gray, cv::COLOR_BGR2GRAY);
    std::vector<std::pair<std::size_t,
      std::vector<cv::Point2f>>> ids_with_features;
    std::size_t n{0};

    // Calculate feature points for trackers.
    for (const auto& tracker: trackers_) {
      BBox bbox_candidate = tracker.second->get_state();
      if (!check_and_resize_state_bbox(bbox_candidate,
          tracker.second->get_state())) {
        // The resized box is too small to receive a matching detection,
        // meaning we can delete the track.
        tracks_to_delete.emplace_back(tracker.first);
        continue;
      }
      std::vector<cv::Point2f> features {
              calculate_feature_points(bbox_candidate, frame_gray)
      };
      ids_with_features.emplace_back(tracker.first, features);
      n += features.size();
    }

    // Delete trackers which moved outside the image space.
    for (const auto& trK_to_del : tracks_to_delete)
      trackers_.erase(trK_to_del);

    // Calculate velocities from optical flow for all tracks.
    std::unordered_map<std::size_t,
      std::vector<struct Velocity_>> trackid_with_velocity;
    if (!ids_with_features.empty())
      trackid_with_velocity =
          calculate_velocities_from_flow(frame_gray, n, ids_with_features);

    // Calculate best velocity for update from mahalanobis distance.
    for (const auto & it : trackid_with_velocity)
      trackers_.at(it.first)->calculate_velocity_for_update(it.second);

    // Update all trackers.
    for (auto & tracker : trackers_)
      tracker.second->update();

    // Create and initialize new trackers for unmatched detections.
    for (std::size_t d : unmatched_dets)
      trackers_.emplace(id_++,
          std::make_unique<FlowBoxTracker>(dets_and_img.detections[d]));

#if !defined(NDEBUG)
    std::cout << "trackers: " << std::endl;
    for (auto& tracker : trackers_)
      std::cout << "tracker " << tracker.first + 1 << " is active with "
                << tracker.second->hits
                << " hits and time_since_update "
                << tracker.second->time_since_update
                << " with a bbox: " << tracker.second->get_state()
                << std::endl;
#endif

    // Return active trackers.
    std::vector<struct Track> active_tracks;
    for (const auto& trk : trackers_) {
      if ((trk.second->time_since_update < max_age_) &&
      (trk.second->hits >= n_init_ || frame_ <= n_init_)) {  // Start on 1st frame.
        struct Track track;
        track.bbox = trk.second->get_state();
        track.id = trk.first + 1;  // +1 as MOT benchmark requires positive.
        active_tracks.emplace_back(track);
      }
    }

    prev_frame_gray_ = frame_gray.clone();
    return active_tracks;
  }

private:
  //! Measurement function if bbox and velocity is provided. Default.
  static const cv::Mat& H_() {
    static const cv::Mat H_ = (cv::Mat_<float>(6, 7) <<
             1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 1, 0);
    return H_;
  }

  //! Measurement function if only a bbox provided.
  static const cv::Mat& H_bbox_only_() {
    static const cv::Mat H_bbox_only = (cv::Mat_<float>(6, 7) <<
            1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0);
    return H_bbox_only;
  }

  //! Measurement function if only velocity is provided.
  static const cv::Mat& H_velocity_only_() {
    static const cv::Mat H_velocity_only = (cv::Mat_<float>(6, 7) <<
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0);
    return H_velocity_only;
  }

  struct Velocity_{
    float vx;
    float vy;
  };

  /**
   * This nested class implements a single target track with state
   * [x, y, s, r, x', y', s'], where x,y are the center, s is the scale/area
   * and r the aspect ratio of the bounding box, and x', y', s' their
   * respective velocities.
   */
  class FlowBoxTracker {
  public:
    /**
     * Constructor.
     *
     * @param initial_bbox A cv::rect_ with the detection in [x, y, w, h]
     *                     format.
     */
    explicit FlowBoxTracker(const BBox& initial_bbox)
            :hits{0},
             time_since_update{0}
    {
      // State transition function F.
      kf_.transitionMatrix = (cv::Mat_<float>(7, 7) <<
              1, 0, 0, 0, 1, 0, 0,
              0, 1, 0, 0, 0, 1, 0,
              0, 0, 1, 0, 0, 0, 1,
              0, 0, 0, 1, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 1);

      // Measurement function H.
      kf_.measurementMatrix = SORTOF::H_();

      // Measurement noise R.
      kf_.measurementNoiseCov = (cv::Mat_<float>(6, 6) <<
              1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0,
              0, 0, 10, 0, 0, 0,
              0, 0, 0, 10, 0, 0,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1);

      // Process noise Q.
      kf_.processNoiseCov = (cv::Mat_<float>(7, 7) <<
              1, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 0, 0,
              0, 0, 0, 0, 0.01, 0, 0,
              0, 0, 0, 0, 0, 0.01, 0,
              0, 0, 0, 0, 0, 0, 0.0001);

      // Initial Conditions, P.
      // Give high uncertainty to the unobservable initial velocities.
      kf_.errorCovPost = (cv::Mat_<float>(7, 7) <<
              10, 0, 0, 0, 0, 0, 0,
              0, 10, 0, 0, 0, 0, 0,
              0, 0, 10, 0, 0, 0, 0,
              0, 0, 0, 10, 0, 0, 0,
              0, 0, 0, 0, 10000, 0, 0,
              0, 0, 0, 0, 0, 10000, 0,
              0, 0, 0, 0, 0, 0, 10000);

      // State is modeled as [x, y, s, r, vx, vy, vs].
      kf_.statePost.at<float>(0, 0) = initial_bbox.x + initial_bbox.width / 2;
      kf_.statePost.at<float>(1, 0) = initial_bbox.y + initial_bbox.height / 2;
      kf_.statePost.at<float>(2, 0) = initial_bbox.area();
      kf_.statePost.at<float>(3, 0) = initial_bbox.width / initial_bbox.height;
      bbox = initial_bbox;
      found_flow = false;
      got_bbox = false;
    }

    ~FlowBoxTracker() = default;

    //! Calculate the velocity for the measurement update step depending on the
    //! Mahalanobis distance.
    void calculate_velocity_for_update(
        const std::vector<struct Velocity_>& velocities) {

      // Create mean vector.
      cv::Mat mean(2, 1, CV_32FC1);
      mean.at<float>(0,0) = kf_.statePre.at<float>(4,0);
      mean.at<float>(1,0) = kf_.statePre.at<float>(5,0);

      // Create inverted covariance matrix.
      cv::Mat S(2, 2, CV_32FC1);
      S.at<float>(0,0) = kf_.errorCovPre.at<float>(4,4);
      S.at<float>(0,1) = kf_.errorCovPre.at<float>(4,5);
      S.at<float>(1,0) = kf_.errorCovPre.at<float>(5,4);
      S.at<float>(1,1) = kf_.errorCovPre.at<float>(5,5);
      cv::Mat S_inv {S.inv(cv::DECOMP_SVD)};

      std::map<double, struct SORTOF::Velocity_> mahalanobisDist_and_velocity;
      for (const auto& v: velocities) {
        cv::Mat x(2, 1, CV_32FC1);
        x.at<float>(0,0) = v.vx;
        x.at<float>(1,0) = v.vy;

        auto dist = cv::Mahalanobis(x, mean, S_inv);
        mahalanobisDist_and_velocity.emplace(dist, v);
      }

      velocity_to_update.vx = mahalanobisDist_and_velocity.begin()->second.vx;
      velocity_to_update.vy = mahalanobisDist_and_velocity.begin()->second.vy;
      found_flow = true;
    }

    //! Return the current bounding box estimate.
    [[nodiscard]] BBox get_state() const
    {
      return x_to_bbox(kf_.statePost);
    }

    //! Advances the state vector on the current time step.
    void predict()
    {
      // https://github.com/abewley/sort/issues/43
      if (kf_.statePost.at<float>(6, 0) + kf_.statePost.at<float>(2, 0) <= 0)
        kf_.statePost.at<float>(6, 0) *= 0.0;
      time_since_update++;
      bbox = x_to_bbox(kf_.predict());
    }

    //! Perform measurement update depending on whether a bounding box and
    //! velocity, only velocity or only a bounding box is provided.
    void update() {
      if (found_flow) {
        if (got_bbox) {
          ++hits;
          time_since_update = 0;
          auto z = bbox_to_z(bbox, velocity_to_update.vx, velocity_to_update.vy);
          kf_.correct(z);
        }
        else if (!got_bbox) {
          kf_.measurementMatrix = H_velocity_only_();
          cv::Mat z{(cv::Mat_<float>({
                  0, 0, 0, 0, velocity_to_update.vx, velocity_to_update.vy
          }))};
          kf_.correct(z);
          kf_.measurementMatrix = H_();
        }
      }
      else if (!found_flow && got_bbox) {
        kf_.measurementMatrix = H_bbox_only_();
        ++hits;
        time_since_update = 0;
        kf_.correct(bbox_to_z(bbox, 0, 0));
        kf_.measurementMatrix = H_();
      }

      found_flow = false;
      got_bbox = false;
    }

    BBox bbox;
    bool found_flow;
    bool got_bbox;
    std::size_t hits;
    std::size_t time_since_update;

  private:
    cv::KalmanFilter kf_ = cv::KalmanFilter(7, 6, 0);
    struct SORTOF::Velocity_ velocity_to_update{};

    //! Convert bounding box in the form
    //! [x, y, width, height, velocity_x, velocity_y]
    //! and return z in the form
    //! [center_x,center_y, scale, ratio, velocity_x, velocity_y].
    static cv::Mat bbox_to_z(
        const BBox& bbox, const float& vx, const float& vy)
    {
      auto center_x{bbox.x + bbox.width / 2};
      auto center_y{bbox.y + bbox.height / 2};
      auto area{bbox.area()};
      auto ratio{bbox.width / bbox.height};

      cv::Mat z{(cv::Mat_<float>({center_x, center_y, area, ratio, vx, vy}))};
      return z;
    }

    //! Convert bounding box in the center form
    //! [center_x, center_y, scale, ratio] and returns it in the form of
    //! [x, y, width, height].
    static BBox x_to_bbox(const cv::Mat& state)
    {
      auto center_x{state.at<float>(0, 0)};
      auto center_y{state.at<float>(1, 0)};
      auto area{state.at<float>(2, 0)};
      auto ratio{state.at<float>(3, 0)};

      auto width{sqrt(area * ratio)};
      auto height{area / width};
      auto x{(center_x - width / 2)};
      auto y{(center_y - height / 2)};

      // BBox in image space.
      if (x < 0 && center_x > 0)
        x = 0;
      if (y < 0 && center_y > 0)
        y = 0;
      return BBox(x, y, width, height);
    }
  }; // class FlowBoxTracker

  void associate_detections_to_trackers(
      const std::vector<BBox >&,
      std::vector<std::size_t>&);

  std::vector<cv::Point2f> calculate_feature_points(
      const BBox& bbox,
      const cv::Mat& frame_gray);

  std::unordered_map<std::size_t, std::vector<struct Velocity_>>
          calculate_velocities_from_flow(
          const cv::Mat& next_img,
          const std::size_t& n,
          const std::vector<std::pair<std::size_t,
          std::vector<cv::Point2f>>>& ids_with_features);

  bool check_and_resize_state_bbox(
      BBox& bbox_candidate, const BBox& bbox_state) const;

  static float iou(const BBox&, const BBox&);

  // Original SORT stuff.
  std::size_t frame_;
  std::size_t id_;
  unsigned int max_age_;
  unsigned int n_init_;
  const double precision_;
  std::unordered_map<std::size_t, std::unique_ptr<FlowBoxTracker>> trackers_;
  double iou_threshold_;

  // Flow stuff.
  cv::Mat prev_frame_gray_;
  cv::Ptr<cv::FastFeatureDetector>fast_;
  cv::Ptr<cv::ORB>orb_;
  cv::Ptr<cv::optflow::RLOFOpticalFlowParameter> rlof_params_;

  // Helper stuff.
  float img_width_{};
  float img_height_{};
  BBox image_space_;
}; // class SORTOF

//! Assign detections to tracked boxes.
void SORTOF::associate_detections_to_trackers(
        const std::vector<BBox >& detections,
        std::vector<std::size_t>& unmatched_dets)
{
  if (trackers_.empty()) {
    for (std::size_t i{0}; i < detections.size(); ++i)
      unmatched_dets.push_back(i);
    return;
  }

  // Create iou cost matrix.
  std::size_t rows {detections.size()};
  std::size_t col;
  std::size_t cols {trackers_.size()};
  dlib::matrix<std::size_t> iou_cost(rows, cols);

  for (std::size_t row{0}; row < rows; ++row) {
    col = 0;
    for (auto& trk : trackers_) {
      iou_cost(row, col) =
          std::size_t(precision_ * iou(detections[row],
              trk.second->get_state()));
      ++col;
    }
  }

  // Create mapping of rows from iou cost matrix to tracker ids.
  col = 0;
  std::vector<std::size_t> idx_to_trkid;
  for (auto& trk : trackers_) {
    idx_to_trkid.push_back(trk.first);
    ++col;
  }

  // Pad iou cost matrix if not square.
  if (iou_cost.nr() > iou_cost.nc())
    iou_cost =
        dlib::join_rows(iou_cost,
            dlib::zeros_matrix<std::size_t>(
                1, iou_cost.nr() - iou_cost.nc()));
  else if (iou_cost.nc() > iou_cost.nr())
    iou_cost =
        dlib::join_cols(iou_cost,
            dlib::zeros_matrix<std::size_t>(
                iou_cost.nc() - iou_cost.nr(), 1));

  // Solve linear assignment problem.
  std::vector<long> lap = dlib::max_cost_assignment(iou_cost);

#if !defined(NDEBUG)
  std::cout << "iou_cost matrix: "<< std::endl;
    std::cout << iou_cost << std::endl;
    std::cout << "assignment: "<< std::endl;
    for (auto & l: lap)
      std::cout << l <<' ' ;
    std::cout << std::endl;
#endif

  // Filter out matched with low iou and assign detections to tracks.
  for (std::size_t d{0}; d < lap.size(); ++d) {
    if (iou_cost(d, lap[d]) < iou_threshold_) {
      if (d < detections.size())
        unmatched_dets.push_back(d);
    } else {
      size_t track_id = idx_to_trkid[lap[d]];
      trackers_.at(track_id)->bbox = detections.at(d);
      trackers_.at(track_id)->got_bbox = true;
    }
  }
}

//! Retrieve feature points from cv::ORB or cv::FAST in the given bounding box.
std::vector<cv::Point2f> SORTOF::calculate_feature_points(
        const BBox& bbox,
        const cv::Mat& frame_gray)
{
  cv::Mat roi = frame_gray(bbox);
  std::vector<cv::KeyPoint> keypoints;

  bool try_fast {false};

  try {
    orb_->detect(roi, keypoints);
  }
  catch (cv::Exception& e_orb){
    try_fast = true;
  }

  // It may possible, that ORB failed.
  if (keypoints.empty()) try_fast = true;

  // Fallback to FAST.
  if (try_fast) {
    try {
      fast_->detect(roi, keypoints);
    }
    catch (cv::Exception& e_fast) {
      return std::vector<cv::Point2f>();
    }
  }

  // It may also possible, that FAST didn't find anything.
  if (keypoints.empty()) return std::vector<cv::Point2f>();

  std::vector<cv::Point2f> p0;
  cv::KeyPoint::convert(keypoints, p0);
  // Convert coordinates back to fit in the original frame.
  for (auto& p : p0) {
    p.x += bbox.x;
    p.y += bbox.y;
  }

  return p0;
}

//! Calculate velocities for a given set of feature points with optical flow.
std::unordered_map<std::size_t, std::vector<struct SORTOF::Velocity_>>
        SORTOF::calculate_velocities_from_flow(
        const cv::Mat& next_img,
        const std::size_t& n,
        const std::vector<std::pair<std::size_t,
        std::vector<cv::Point2f>>>& ids_with_features)
{
  std::vector<cv::Point2f> p0;
  p0.reserve(n);
  std::vector<std::size_t> ids;
  ids.reserve(n);

  // Put all points in an array and all corresponding ids in a second
  // array of the same size.
  for (const auto& trk : ids_with_features) {
    if (trk.second.empty())
      continue; // No feature points for this track id were found.
    for (const auto& pnts : trk.second) {
      p0.push_back(pnts);
      ids.push_back(trk.first);
    }
  }

  std::unordered_map<std::size_t,
  std::vector<struct SORTOF::Velocity_>> trackId_and_velocities{};

  // It may be possible that no features were found at all.
  if (p0.empty()) return trackId_and_velocities;

  std::vector<cv::Point2f> p1;
  p1.reserve(n);
  std::vector<uchar> status;
  status.reserve(n);
  std::vector<float> err;
  err.reserve(n);

  cv::optflow::calcOpticalFlowSparseRLOF(
      next_img,
      prev_frame_gray_,
      p0,
      p1,
      status,
      err,
      rlof_params_);

  // Update points and ids for the ones optical flow was found.
  std::vector<std::size_t> valid_ids;
  std::vector<cv::Point2f> valid_p0;
  std::vector<cv::Point2f> valid_p1;
  for (std::size_t i =0; i < p0.size(); i++) {
    if (status[i] == 1) {
      valid_p0.push_back(p0[i]);
      valid_p1.push_back(p1[i]);
      valid_ids.push_back(ids[i]);
    }
  }

  // In case no flow at all was found we can skip here.
  if (!valid_ids.empty()) {
    for (std::size_t i {0}; i < valid_ids.size(); ++i) {
      // Calculate velocity for the point.
      struct SORTOF::Velocity_ velocity{};
      velocity.vx = p0[i].x - p1[i].x;
      velocity.vy = p0[i].y - p1[i].y;
      auto it = trackId_and_velocities.find(valid_ids[i]);
      // If the id is already in the map, add the velocity struct to the
      // respective vector, else create new entry in the map.
      if (it != trackId_and_velocities.end())
        it->second.emplace_back(velocity);
      else {
        std::vector<struct SORTOF::Velocity_> tmp;
        tmp.emplace_back(velocity);
        trackId_and_velocities.emplace(valid_ids[i], tmp);
      }
    }
  }
  return trackId_and_velocities;
}

//! Check if the state of a track at the current time step is 100% in the
//! frame. If not, resize it so it will be in image space. If the resized
//! bounding box is too small to get a matching detection within the given iou
//! threshold, the track is going to be deleted.
bool SORTOF::check_and_resize_state_bbox(
    BBox& bbox_candidate,
    const BBox& bbox_state
) const
{
  // Check if bbox is still in images space.
  if ((bbox_candidate & SORTOF::image_space_) == bbox_candidate) return true;

  // Resize if not.
  if (bbox_candidate.x < 0) bbox_candidate.x = 0;
  if (bbox_candidate.y < 0) bbox_candidate.y = 0;

  if (bbox_candidate.x + bbox_candidate.width > SORTOF::img_width_)
    bbox_candidate.width = SORTOF::img_width_ - bbox_candidate.x;
  if (bbox_candidate.y + bbox_candidate.height > SORTOF::img_height_)
    bbox_candidate.height = SORTOF::img_height_ - bbox_candidate.y;

  // Get iou of resized bbox and states bbox.
  auto iou {bbox_candidate.area() / bbox_state.area()};
  if (iou * SORTOF::precision_ < SORTOF::iou_threshold_)
    return false;
  return true;
}

//! Compute intersection-over-union between two bounding boxes.
float SORTOF::iou(const BBox& det, const BBox& trk)
{
  auto xx1{std::max(det.tl().x, trk.tl().x)};
  auto yy1{std::max(det.tl().y, trk.tl().y)};
  auto xx2{std::min(det.br().x, trk.br().x)};
  auto yy2{std::min(det.br().y, trk.br().y)};
  auto width{std::max(0.f, xx2 - xx1)};
  auto height{std::max(0.f, yy2 - yy1)};
  auto intersection{width * height};
  float union_area{det.area() + trk.area() - intersection};
  return intersection / union_area;
}
} // namespace sort_of
#endif
