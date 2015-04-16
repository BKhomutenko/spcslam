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

void testBins(const StereoSystem & stereo);

void testBruteForce();

void testStereoMatch(const StereoSystem & stereo, const vector<testPoint> & cloud);

void testMatchReprojected();


#endif
