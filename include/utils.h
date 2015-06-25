#ifndef _SPCMAP_UTILS_H_
#define _SPCMAP_UTILS_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "extractor.h"

void drawPoints(const vector<Eigen::Vector2d> & ptVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out);

void drawPoints(const vector<Feature> & fVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out);

void drawPoints(const vector<Feature> & fVec1,
                const vector<Feature> & fVec2,
                cv::Mat & out, cv::Scalar color1, cv::Scalar color2);

#endif
