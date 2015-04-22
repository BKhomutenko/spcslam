#ifndef _SPCSLAM_MATCHER_TEST_H_
#define _SPCSLAM_MATCHER_TEST_H_

#include "Eigen/Eigen"

#include "vision.h"

struct testPoint
{
    Eigen::Vector3d pt;
    Eigen::Matrix<float,64,1> desc;

    float size;
    float angle;
};

void testMatching();
void testBruteForce();
void testStereoMatch();
void testMatchReprojected();

void displayBruteForce();
void displayBins(const StereoSystem & stereo);
void displayStereoMatch(const StereoSystem & stereo);
void displayStereoMatch_2(const StereoSystem & stereo);

StereoSystem initVirtualStereo();
vector<testPoint> initCloud();

#endif
