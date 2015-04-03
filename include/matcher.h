#ifndef _MATCHER_H_
#define _MATCHER_H_

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

#include "extractor.h"
#include "vision.h"

using namespace std;

class Matcher
{
public:

    Eigen::MatrixXi binMapL;
    Eigen::MatrixXi binMapR;

    void bruteForce(const vector<Feature> & kpVec1, const vector<Feature> & kpVec2, vector<int> & matches);

    void stereoMatch(const vector<Feature> & kpVec1, const vector<Feature> & kpVec2, vector<int> & matches);

    void matchReprojected(const vector<Feature> & kpVec1, const vector<Feature> & kpVec2, vector<int> & matches);

    void initStereoBins(const StereoSystem & stereo);

};

#endif
