#include "extractor.h"
#include "matcher.h"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp> 

void Extractor::operator()(const cv::Mat & img, std::vector<KeyPoint> & kpVec)
{
  
    vector<cv::KeyPoint> cvKpVec;
    cv::Mat descriptors;

    det.detect(img, cvKpVec);
    for (auto & kp : cvKpVec)
    {
        kp.angle = -1;
    } 

    extr.compute(img, cvKpVec, descriptors);

    int N = cvKpVec.size();
    kpVec.clear(); 

    for (int i = 0; i < N; i++)
    {    
        const cv::KeyPoint & cvkp = cvKpVec[i];
        float * ptr = (float *)descriptors.row(i).data;
        KeyPoint kp(cvkp.pt.x, cvkp.pt.y, ptr, cvkp.size, cvkp.angle);
        kpVec.push_back(kp);        
    }
  
}
