## SORT with Optical Flow

Extending SORT's Kalman filter measurement model to add a velocity component 
derived from Optical Flow. This was part of my Bachelor's Thesis for Computer
Science in collaboration with the German Aerospace Center. Although not used for
pedestrian tracking, the method can be evaluated and used with the 
[MOT](https://motchallenge.net) datasets.

Basic idea can be summarized as:
- Get a region of interest in form of a bounding box and detect feature points
  in this region.
- Calculate the optical flow for the detected feature points to the frame 
  before, providing a velocity.
- Choose the velocity for updating based upon the lowest Mahalanobis distance.
- Update the Kalman filter of the track with the detection and/or the velocity.

For the detection of feature points I use OpenCV's 
[ORB](https://docs.opencv.org/4.5.0/db/d95/classcv_1_1ORB.html), calculation of
the sparse optical flow is done by OpenCV's 
[implementation](https://docs.opencv.org/4.5.0/d2/d84/group__optflow.html#ga32ba4b0f6a21684d99d45f6bc470f480)
of the Robust Local Optical Flow (RLOF).

This implementation runs at ~50FPS on the
[2D MOT 2015](https://motchallenge.net/data/2D_MOT_2015/) Dataset on an AMD 
Ryzen7 3700X with 64GB RAM with up to 50 feature points per detection,
[parametrization](https://docs.opencv.org/4.5.0/d4/d91/classcv_1_1optflow_1_1RLOFOpticalFlowParameter.html)
for RLOF is set to:
```C++
useIlluminationModel = false;
useInitialFlow = false;
maxIteration = 5;
setUseMEstimator(false);
setUseGlobalMotionPrior(true);
setSupportRegionType(cv::optflow::SR_FIXED);
```

### Dependencies

The following external libraries are needed:
- [OpenCV](https://www.opencv.org) for the Kalman filter, the bounding box 
  representation, calculation of feature points and calculation of optical 
  flow.
- [dlib](http://www.dlib.net) for the linear assignment problem.

If you are on an ArchLinux based system like me, install OpenCV from the
official repository:
```
$ sudo pacman -S opencv glu glew hdf5 vtk
```
dlib can be obtained from the [aur](https://aur.archlinux.org/packages/dlib/).

On Ubuntu based distributions simply install them with:
```
$ sudo apt-get install libopencv-contrib-dev libdlib-dev
```

### Build

I use cmake for the build process.

```
$ git clone https://github.com/tylernewnoise/SORT_OF
$ cd SORT_OF
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make
```

### Demo

Put your data in this folder structure:
```
SORT_OF
  └─ data
    └─ Sequence
      ├─ det
      └─ gt
```
Run
```
$./SORT_OF -d
```
inside the ``SORT_OF`` directory to get a visualization with neat bounding 
boxes. Omit the ``-d`` to run the tracker offline. Add ``-o`` to write results
to ``output/SequenceName.txt`` in mot challenge format. One can use 
[py-motmetrics](https://github.com/cheind/py-motmetrics) to evaluate the 
results.

I tested and wrote this on AchLinux, for other operating system changes may
have to be made.

### Usage

Functionality is written in a single header file for simple usage in other 
projects, just import the header with:
```C++
#import sort_of.h

```
Detections and the corresponding image have to be passed as a 
``struct DetectionsAndImg{}`` which is defined as:
```c++
struct DetectionsAndImg {
  std::vector<BBox> detections;
  cv::Mat img;
};
```
Tracks are returned as ``std::vector<struct Track>``, whereat the 
``struct Track{}`` is defined as:
```c++
struct Track {
  BBox bbox;
  std::size_t id{};
};
```
See the 
[demo.cpp](https://github.com/tylernewnoise/SORT_with_Optical_Flow/blob/main/src/demo.cpp#L133)
for an example.

### Results

I used Alex's [detections](https://github.com/abewley/sort/tree/master/data/train)
and [py-motetrics](https://github.com/cheind/py-motmetrics) for evaluation.
These are the data sets from a static camera only. Results can be improved by
increasing the value of 
[max_age](https://github.com/tylernewnoise/SORT_with_Optical_Flow/blob/main/src/demo.cpp#L129).

Sequence       | Rcll  |  Prcn |  GT |  MT | PT |   ML |   FP |   FN | IDs |   FM | MOTA  |  MOTP
-------------- |:-----:|:-----:|:---:|:---:|:---:|:---:|:----:|:----:|:---:|:----:|:-----:|:------
ADL-Rundle-6   | 58.2% | 76.0% |  24 |   6 |  16 |   2 |  921 | 2095 |  67 |  102 | 38.5% | 74.7%
TUD-Stadtmitte | 75.2% | 97.5% |  10 |   6 |   4 |   0 |   22 |  287 |  10 |   16 | 72.4% | 75.3%
KITTI-17       | 69.3% | 91.7% |   9 |   1 |   8 |   0 |   43 |  210 |   8 |   17 | 61.8% | 71.8%
Venice-2       | 43.2% | 64.2% |  26 |   8 |  11 |   7 | 1723 | 4057 |  61 |  115 | 18.2% | 73.6%
PETS09-S2L1    | 77.1% | 87.3% |  19 |   9 |  10 |   0 |  502 | 1025 |  96 |  184 | 63.7% | 67.8%
TUD-Campus     | 68.0% | 92.8% |   8 |   4 |   4 |   0 |   19 |  115 |   5 |   13 | 61.3% | 73.6%
**OVERALL**    | 58.6% | 77.4% |  96 |  34 |  53 |   9 | 3230 | 7789 | 247 |  447 | 40.2% | 72.1%

### Resources

SORT paper, describing the original Simple Online and Realtime Tracking method:

```
@inproceedings{sort2016,
    author = {Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
    booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
    title = {Simple online and realtime tracking},
    year = {2016},
    pages = {3464-3468}
}
```
Also checkout [nwojke](https://github.com/nwojke) 's
[DeepSORT](https://github.com/nwojke/deepsort) implementation.

ORB feature point detection is explained on 
[OpenCV's site](https://docs.opencv.org/4.5.0/d1/d89/tutorial_py_orb.html).

Papers for the Robust Optical Flow can be found 
[here](https://www.nue.tu-berlin.de/menue/forschung/forschungsgebiete/multimedia_analyse_und_verarbeitung/rlof00/).

Other SORT C++ implementations:
- [samuelmurray](https://github.com/samuelmurray/tracking-by-detection)
- [yasenh](https://github.com/yasenh/sort-cpp)
- [mcxmiming](https://github.com/mcximing/sort-cpp)
- [my own](https://github.com/tylernewnoise/sort_in_cpp)
