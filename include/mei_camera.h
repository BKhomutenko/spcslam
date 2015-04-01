#ifndef _SPCMAP_MEI_H_
#define _SPCMAP_MEI_H_

//OpenCV
#include <opencv2/opencv.hpp>

//Eigen
#include <Eigen/Eigen>

#include "geometry.h"

using namespace cv;
using namespace Eigen;

class MeiCamera : public Camera
{
public:
    unsigned int imageWidth;
    unsigned int imageHeight;
    double xi;
    double fu;
    double fv;
    double u0;
    double v0; 

     /// takes raw image points and apply undistortion model to them
    void reconstructPoints(const Point2f & src, Point2f & dst) const;
    
    /// projects 3D points onto the original image
    void projectPoint(const Point3f & src, Point2f & dst) const;
};

#endif
