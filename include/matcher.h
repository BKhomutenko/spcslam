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

    double bfDistTh = 0.20;
    double bfDistTh2 = 0.20;
    double stereoDistTh = 0.20;
    double reproDistTh = 0.20;

    const int bf2matches = 3;

    double alfaTolerance = 0.4; //degrees
    double betaTolerance = 0.4; //degrees

    double binDelta = 1; //degrees

    Eigen::MatrixXi binMapL;
    Eigen::MatrixXi binMapR;

    Eigen::MatrixXd alfaMap1;
    Eigen::MatrixXd betaMap1;
    Eigen::MatrixXd alfaMap2;
    Eigen::MatrixXd betaMap2;

    void bruteForce(const vector<Feature> & featuresVec1,
                    const vector<Feature> & featuresVec2,
                    vector<int> & matches) const;

    void bruteForceOneToOne(const vector<Feature> & featuresVec1,
                            const vector<Feature> & featuresVec2,
                            vector<int> & matches) const;

    void bruteForce_2(const vector<Feature> & featuresVec1,
                      const vector<Feature> & featuresVec2,
                      vector<vector<int>> & matches) const;

    void stereoMatch(const vector<Feature> & featuresVec1,
                     const vector<Feature> & featuresVec2,
                     vector<int> & matches) const;

    void stereoMatch_2(const vector<Feature> & featuresVec1,
                     const vector<Feature> & featuresVec2,
                     vector<int> & matches) const;

    void matchReprojected(const vector<Feature> & featuresVec1,
                          const vector<Feature> & featuresVec2,
                          vector<int> & matches, double radius) const;

    void initStereoBins(const StereoSystem & stereo);

    void computeMaps(const StereoSystem & stereo);

};

#endif
