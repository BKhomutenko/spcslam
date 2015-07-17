#ifndef _SPCMAP_MATCHER_H_
#define _SPCMAP_MATCHER_H_

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

#include "extractor.h"
#include "vision.h"

using namespace std;

double computeDist(const Descriptor & d1, const Descriptor & d2);

class Matcher
{
public:
    const double thresh2 = 25000; //FIXME
    double bfDistTh = thresh2;
    double bfDistTh2 = thresh2;
    double stereoDistTh = thresh2;
    double reproDistTh = thresh2;

    const int bf2matches = 3;

    double alfaTolerance = 0.4; //degrees
    double betaTolerance = 0.4; //degrees

    double binDelta = 0.5; //degrees

    Eigen::MatrixXi binMapL;
    Eigen::MatrixXi binMapR;

    Eigen::MatrixXd alfaMap1;
    Eigen::MatrixXd betaMap1;
    Eigen::MatrixXd alfaMap2;
    Eigen::MatrixXd betaMap2;

    void bruteForce(const vector<Feature> & featureVec1,
                    const vector<Feature> & featureVec2,
                    vector<int> & matches) const;

    void bruteForceOneToOne(const vector<Feature> & featureVec1,
                            const vector<Feature> & featureVec2,
                            vector<int> & matches) const;

    void bruteForce_2(const vector<Feature> & featureVec1,
                      const vector<Feature> & featureVec2,
                      vector<vector<int>> & matches) const;

    void stereoMatch(const vector<Feature> & featureVec1,
                     const vector<Feature> & featureVec2,
                     vector<int> & matches) const;

    void stereoMatch_2(const vector<Feature> & featureVec1,
                     const vector<Feature> & featureVec2,
                     vector<int> & matches) const;

    void matchReprojected(const vector<Feature> & featureVec1,
                          const vector<Feature> & featureVec2,
                          vector<int> & matches, double radius) const;

    void initStereoBins(const StereoSystem & stereo);

    void computeMaps(const StereoSystem & stereo);

};

#endif
