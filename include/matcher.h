#ifndef _SPCMAP_MATCHER_H_
#define _SPCMAP_MATCHER_H_

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

    enum BFType { simple, oneToOne };

    double bfDistTh = 1000;
    double binDelta = 1; //degrees

    Eigen::MatrixXi binMapL;
    Eigen::MatrixXi binMapR;

    void bruteForce(const vector<Feature> & fVec1,
                    const vector<Feature> & fVec2,
                    vector<int> & matches);

    void bruteForceOneToOne(const vector<Feature> & fVec1,
                            const vector<Feature> & fVec2,
                            vector<int> & matches);

    void stereoMatch(const vector<Feature> & fVec1,
                     const vector<Feature> & fVec2,
                     vector<int> & matches);

    void matchReprojected(const vector<Feature> & fVec1,
                          const vector<Feature> & fVec2,
                          vector<int> & matches);

    void initStereoBins(const StereoSystem & stereo);

    Extractor extractor;

};

#endif
