#ifndef _EXTRACTOR_H_ 
#define _EXTRACTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp> 

#include "matcher.h"

class Extractor
{

  cv::SurfFeatureDetector det;
  cv::SurfDescriptorExtractor extr;

public:
  
  //Extractor() {}
  
  Extractor(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, bool upright)
  : det(hessianThreshold, nOctaves, nOctaveLayers, extended, upright) {}
  
  void operator()(const cv::Mat & img, std::vector<KeyPoint> & kpVec);
  
};

#endif