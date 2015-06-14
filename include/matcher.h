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

    double bfDistTh = 0.15;
    double binDelta = 1; //degrees

    Eigen::MatrixXi binMapL;
    Eigen::MatrixXi binMapR;

    Eigen::MatrixXd alfaMap1;
    Eigen::MatrixXd betaMap1;
    Eigen::MatrixXd alfaMap2;
    Eigen::MatrixXd betaMap2;

    void bruteForce(const vector<Feature> & fVec1,
                    const vector<Feature> & fVec2,
                    vector<int> & matches) const;

    void bruteForceOneToOne(const vector<Feature> & fVec1,
                            const vector<Feature> & fVec2,
                            vector<int> & matches) const;

    void stereoMatch(const vector<Feature> & fVec1,
                     const vector<Feature> & fVec2,
                     vector<int> & matches) const;

    void matchReprojected(const vector<Feature> & fVec1,
                          const vector<Feature> & fVec2,
                          vector<int> & matches) const;

    void initStereoBins(const StereoSystem & stereo);

    void computeMaps(const StereoSystem & stereo);

};

#endif
