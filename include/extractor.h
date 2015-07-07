#ifndef _EXTRACTOR_H_
#define _EXTRACTOR_H_

#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

using Eigen::Vector2d;
using Eigen::Matrix;

using namespace std;

enum FeatureType {SURF, Custom};

struct Feature
{

    Vector2d pt; // representation: (x, y). TODO: change representation to (y, x) ?
    Matrix<float,64,1> desc;

    float size, angle;

    Feature(double x, double y, const Matrix<float,64,1> & d, float size, float angle)
                : pt(x, y) , desc(d) , size(size) , angle(angle) {}

    Feature(const Vector2d & p, const Matrix<float,64,1> & d)
                : pt(p) , desc(d) {}

    Feature(const Vector2d & p, const Matrix<float,64,1> & d, float size, float angle)
                : pt(p) , desc(d), size(size), angle(angle) {}

    Feature(double x, double y, float * d)
                : pt(x, y) , desc((float *) d) {}

    Feature(double x, double y, float * d, float size, float angle)
                : pt(x, y) , desc((float *) d), size(size), angle(angle) {}

    Feature() {}
};

class Extractor
{
private:

    cv::SurfFeatureDetector det;
    cv::SurfDescriptorExtractor extr;

    FeatureType fType = FeatureType::SURF;

    cv::Mat kernel;
    cv::Mat mask;
    int thresh = 50;
    const int descWidth = 4;

    vector<Eigen::MatrixXi> binMaps;
    const int nDivVertical = 4;
    const int nDivHorizontal = 6;

    const vector<vector<double> > circle1 = { { 680, 462, 632 },
                                              { 1253, 328, 1092 } }; // circle definition: { u0, v0, r }
    const vector<vector<double> > circle2 = { { 292, 382, 850 },
                                              { 324, 502, 918 } };
    const vector<vector<double> > uBounds = { { 48, 1142 },
                                              { 161, 1242 } };
    const vector<double> vBounds = { 100, 616 };

    const int nFeatures = 300;
    vector<vector<int>> featureDistributions;

public:

    //Extractor() {}

    Extractor(double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended, bool upright)
    : det(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
    {
        computeMaps();
    }

    Extractor()
    {
        computeMaps();
    }

    void setType(FeatureType featType);

    void operator()(const cv::Mat & img, std::vector<Feature> & featuresVec, int camId);

    void extractFeatures(cv::Mat src, vector<cv::KeyPoint> points, cv::Mat & descriptors);

    void extractFeatures(const vector<cv::Mat> & images, vector< vector<cv::KeyPoint> > & keypoints,
                         vector<cv::Mat> & descriptors);

    void extractDescriptor(cv::Mat src, cv::Point2f pt, int patchSize, cv::Size descSize, cv::Mat & dst);

    void findMax(cv::Mat src, vector<cv::Point2f> & maxPoints, float threshold, int camId);

    void findFeatures(cv::Mat src, std::vector<cv::KeyPoint> & points, int camId, float scale1,
                      float scale2=-1, int steps=3);

    void computeResponse(const cv::Mat src, cv::Mat & dst, float scale);

    void cvtNormimagesmage(cv::Mat & image);

    //void computeBinMaps();

    void computeMaps();

};

#endif
