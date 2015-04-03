#ifndef _EXTRACTOR_H_
#define _EXTRACTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

struct Feature
{

    Eigen::Vector2d pt; // representation: (x, y). TODO: change representation to (y, x) ?
    Eigen::Matrix<float,64,1> desc;

    float size, angle;

    Feature(double x, double y, float * d): pt(x, y) , desc((float *) d) {}
    Feature(double x, double y, float * d, float size, float angle)
                : pt(x, y) , desc((float *) d), size(size), angle(angle) {}
};

class Extractor
{
private:
    cv::SurfFeatureDetector det;
    cv::SurfDescriptorExtractor extr;

public:

    //Extractor() {}

    Extractor(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, bool upright)
    : det(hessianThreshold, nOctaves, nOctaveLayers, extended, upright) {}

    void operator()(const cv::Mat & img, std::vector<Feature> & kpVec);

};

#endif
