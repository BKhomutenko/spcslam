#ifndef _CARTOGRAPHY_TESTS_H_
#define _CARTOGRAPHY_TESTS_H_

#include <vector>
#include <Eigen/Eigen>

using namespace std;

void compare(const vector<Eigen::Vector3d> cloud1, const vector<Eigen::Vector3d> cloud2);

void testGeometry();

void testVision();

void testMei();

void testOdometry();

void testBundleAdjustment();

void testCartography();

#endif
