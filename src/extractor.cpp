#include "extractor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

using namespace std;

void Extractor::operator()(const cv::Mat & img, std::vector<Feature> & featuresVec, int camId)
{

    std::vector<cv::KeyPoint> cvKpVec;
    cv::Mat descriptors;

    /*switch (fType)
    {
        case FeatureType::SURF:
        {
            det.detect(img, cvKpVec);
        }
        case FeatureType::Custom:
        {
            findFeatures(img, cvKpVec, camId, 3);
        }
    }*/

    det.detect(img, cvKpVec);

    //findFeatures(img, cvKpVec, camId, 3);

    /*cv::Mat mask;
    if (not mask.data)
    {
        mask = cv::Mat(img.size(), CV_8U);
        mask(cv::Rect(100, 100, img.cols - 100, img.rows - 300)).setTo(1);
    }
    cv::ORB orb(1000);
    orb(img, mask, cvKpVec, cv::noArray());
*/
    /*for (auto & kp : cvKpVec)
     *    {
     *        kp.angle = -1;
}*/

    extr.compute(img, cvKpVec, descriptors);

    int N = cvKpVec.size();
    featuresVec.clear();

    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint & cvkp = cvKpVec[i];
        float * ptr = (float *)descriptors.row(i).data;
        Feature kp(cvkp.pt.x, cvkp.pt.y, ptr, cvkp.size, cvkp.angle);
        featuresVec.push_back(kp);
    }

}

void Extractor::setType(FeatureType featType)
{
    fType = featType;
}

void Extractor::extractFeatures(const vector<cv::Mat> & images, vector< vector<cv::KeyPoint> > & keypoints,
                                vector<cv::Mat> & descriptors)
{
    if (not mask.data)
    {
        mask = cv::Mat(images[0].size(), CV_8U);
        mask(cv::Rect(100, 100, images[0].cols - 100, images[0].rows - 300)).setTo(1);
    }
    int num = images.size();
    int minHessian = 500; //6500 for surf
    cv::SiftFeatureDetector detector(minHessian);//, 1, 3);
    //Ptr<FeatureDetector> detector = FeatureDetector::create("FAST");
    cv::SiftDescriptorExtractor extractor;
    cv::ORB orb(1000);
    keypoints.assign(num, vector<cv::KeyPoint>());
    descriptors.assign(num, cv::Mat());
    cout << "extracting" << endl;
    cv::Mat outImage;
    //findFeatures(images, keypoints, 3, -1, 1000);
    for (int i = 0; i < num; i++)
    {
        clock_t begin = clock();




        //cout << "." << flush;
        //cvtNormimagesmage(images[i]);
        //detector.detect(images[i], keypoints[i], mask);
        //FAST(images[i], keypoints[i], thresh);

        //findFeatures(images[i], keypoints[i], 3);
        //orb(images[i], mask, keypoints[i], descriptors[i], true);
        //print3(descriptors[i].type(), CV_8U, CV_32S);
        /*if (keypoints[i].size() > 600)
         *                        thresh += 2;
         *                else
         *                        thresh -= 2;*/
        //
        /*drawKeypoints(images[i], keypoints[i], outImage, Scalar::all(-1), 4);
         *                imshow("kp", outImage);
         *                waitKey();*/
        //extractor.compute(images[i], keypoints[i], descriptors[i]);
        //extractFeatures(images[i], keypoints[i], descriptors[i]);
        /*drawCircles(images[i], keypoints[i]);
         *                imshow("feat", images[i]);
         *                waitKey();*/
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        int ms = int(1000.0*elapsed_secs);
        //print2(keypoints[i].size(), ms);
        /*images[i].copyTo(outImage);
         *                drawCircles(outImage, keypoints[i], 5);
         *                boost::format outStr("/home/bogdan/catkin_ws/feat%s_%i.png");
         *                imwrite(boost::str(outStr % "SIFT" % i), outImage);*/
    }
}

void Extractor::cvtNormimagesmage(cv::Mat & image)
{
    image.convertTo(image, CV_32F);
    log(image/10 + 1, image);
    cv::Mat mean, stddev;
    meanStdDev(image, mean, stddev);
    image -= mean.at<double>(0, 0);
    image /= stddev.at<double>(0, 0);
    image *= 120;
    image += 128;
    image.convertTo(image, CV_8U);
}

void Extractor::computeResponse(const cv::Mat src, cv::Mat & dst, float scale)
{
    /*if (src.type() != CV_32F)
     *        {
     *                Mat tmp;
     *                src.convertTo(tmp, CV_32F);
     *                src = tmp;
}*/
    cv::Mat I, Ix, Iy, Ixx, Ixy, Iyy, Ib;

    //Laplacian(src, dst, CV_32F, 5);
    //dst = abs(dst);

    cv::GaussianBlur(src, I, cv::Size(0, 0), 1);
    cv::Sobel(I, Ix, CV_32F, 1, 0, 3, 1/8.);
    cv::Sobel(I, Iy, CV_32F, 0, 1, 3, 1/8.);
    Ixx = Ix.mul(Ix);
    Ixy = Iy.mul(Ix);
    Iyy = Iy.mul(Iy);
    cv::GaussianBlur(Ixx, Ixx, cv::Size(0, 0), scale);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(0, 0), scale);
    cv::GaussianBlur(Iyy, Iyy, cv::Size(0, 0), scale);
    cv::Mat trace = (Ixx + Iyy + 1);
    dst = (Ixx.mul(Iyy) - Ixy.mul(Ixy)) / trace;

    /* cv::imshow("resp", dst);
     *        cv::waitKey();
     *    cv::GaussianBlur(dst, I, cv::Size(0, 0), 1);*/

    dst /=( dst+.1);
    /*cv::imshow("resp", dst);
     *        cv::waitKey();
     *    cv::GaussianBlur(dst, Ixx, cv::Size(0, 0), 5);*/
    /* dst -= Ixx*0.5;
     *    cv::imshow("resp", dst);
     *        cv::waitKey();*/
}

//TODO extend scale to vector<float> scales
void Extractor::findFeatures(cv::Mat src, std::vector<cv::KeyPoint> & points, int camId,
                             float scale1, float scale2, int steps)
{
    if (scale2 == -1)
        scale2 = scale1;
    float step = (scale2 - scale1) / steps;
    if (step < 0.1f)
        step = 0.1f;
    points.clear();
    for (float scale = scale1; scale <= scale2; scale += step)
    {

        cv::Mat resp;
        computeResponse(src, resp, scale);

        vector<cv::Point2f> maxPoints;
        findMax(resp, maxPoints, 0.3, camId);


        for (int i = 0; i < maxPoints.size(); i++)
        {
            points.push_back(cv::KeyPoint(maxPoints[i].x, maxPoints[i].y, (scale*0.5f+4)));
        }
    }
}

void Extractor::findMax(cv::Mat src, vector<cv::Point2f> & maxPoints, float threshold, int camId)
{
    assert(src.type() == CV_32F);
    assert(sizeof(float) == 4);
    int cols = src.cols;
    float * srcPtr = (float *)(src.data);
    int neiu[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int neiv[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    vector<vector<vector<double> > > dataVec;
    vector<vector<cv::Point2f> > pointsVec;

    for (int i = 0; i < nDivisions*2; i++)
    {
        vector<cv::Point2f> vecP;
        pointsVec.push_back(vecP);
        vector<vector<double>> vecD;
        dataVec.push_back(vecD);
    }

    for (int v = 0; v < 966; v++)
    {
        for (int u = 0; u < 1296; u++)
        {
            int bin = binMaps.at(camId-1)(u, v);
            if (bin == 0)
                continue;
            float srcVal = srcPtr[v*cols + u];
            if (srcVal < threshold)
                continue;
            bool isMax = true;
            for (int i = 0; i < 8; i++)
            {
                if(srcVal <= srcPtr[(v + neiv[i])*cols + u + neiu[i]])
                {
                    isMax = false;
                    break;
                }
            }
            if (isMax)
            {
                cv::Point2f maxPt(u, v);
                pointsVec[bin-1].push_back(maxPt);
                vector<double> pointData = { (double)pointsVec[bin-1].size()-1, (double)srcVal };
                dataVec[bin-1].push_back(pointData);
            }
        }
    }

    for (int i = 0; i < nDivisions*2; i++)
    {
        sort(dataVec[i].begin(), dataVec[i].end(),
            [](const vector<double>& a, const vector<double>& b)
            { return a[1] > b[1]; });
    }

    for (int i = 0; i < nDivisions*2; i++)
    {
        int j = 0;
        while (j < min((double)featureDistributions[camId-1][i], (double)dataVec[i].size()))
        {
            maxPoints.push_back(pointsVec[i][dataVec[i][j][0]]);
            j++;
        }
    }
}


// void Extractor::findMax(cv::Mat src, vector<cv::Point2f> & maxPoints, float threshold)
// {
//     assert(src.type() == CV_32F);
//     assert(sizeof(float) == 4);
//     int cols = src.cols;
//     float * srcPtr = (float *)(src.data);
//     int neiu[] = {-1, -1, -1, 0, 0, 1, 1, 1};
//     int neiv[] = {-1, 0, 1, -1, 1, -1, 0, 1};
//     for (int v = 0; v < src.rows; v++)
//     {
//         for (int u = 0; u < src.cols; u++)
//         {
//             //TODO check the mask
//             if ( (v - 400)*(v-400)/1.5 + (u- src.cols/2)*(u - src.cols/2)/3.5 > 320*320)
//                 continue;
//             bool isMax = true;
//             float srcVal = srcPtr[v*cols + u];
//             if (srcVal < threshold)
//                 continue;
//             for (int i = 0; i < 8; i++)
//             {
//                 if(srcVal <= srcPtr[(v + neiv[i])*cols + u + neiu[i]])
//                 {
//                     isMax = false;
//                     break;
//                 }
//             }
//             if (isMax == false)
//             {
//                 continue;
//             }
//             cv::Point2f maxPt(u, v);
//             maxPoints.push_back(maxPt);
//         }
//     }
// }

void Extractor::extractDescriptor(cv::Mat src, cv::Point2f pt, int patchSize, cv::Size descSize, cv::Mat & dst)
{
    cv::Mat patch;
    getRectSubPix(src, cv::Size(patchSize, patchSize), pt, patch, CV_32F);
    resize(patch, patch, descSize);
    patch = patch.mul(kernel);
    patch = patch.reshape(0, 1);
    cv::Mat mean, stddev;
    /*meanStdDev(patch, mean, stddev);
     *        patch -= mean.at<double>(0, 0);
     *        patch /= stddev.at<double>(0, 0);*/
    patch.copyTo(dst);
}

void Extractor::extractFeatures(cv::Mat src, vector<cv::KeyPoint> points, cv::Mat & descriptors)
{
    kernel = cv::getGaussianKernel(descWidth, descWidth/2, CV_32F);
    kernel = kernel * kernel.t();
    descriptors = cv::Mat(points.size(), descWidth*descWidth, CV_32F);
    cv::Size descSize = cv::Size(descWidth, descWidth);

    for (int i = 0; i < points.size(); i++)
    {
        cv::Mat row = descriptors.row(i);
        int patchSize = points[i].size * 2;

        extractDescriptor(src, points[i].pt, points[i].size, descSize, row);

    }
}

void Extractor::computeBinMaps()
{
    int width = 1296;
    int height = 966;

    for (int map = 0; map <= 1; map++)
    {
        Eigen::MatrixXi m;
        m.resize(width, height);
        binMaps.push_back(m);

        double ellipseA = sqrt(ellipseAsq);
        double binWidth = ellipseA*2 / (double)nDivisions;
        vector<int> ranges(nDivisions + 1);
        vector<int> binArea(nDivisions*2, 0);
        int ellipseArea = 0;

        for (int i = 0; i < nDivisions + 1; i++)
        {
            ranges[i] = ellipseU0[map] - ellipseA + binWidth * i;
        }

        for (int u = 0; u < width; u++)
        {
            for (int v = 0; v < height; v++)
            {
                int indexVertical = 0;
                int indexHorizontal = 1;
                if ((u - ellipseU0[map])*(u - ellipseU0[map])/ellipseAsq +
                    (v - ellipseV0)*(v - ellipseV0)/ellipseBsq <= 1)
                {
                    for (int i = 0; i < nDivisions; i++)
                    {
                        if (u > ranges[i] and u <= ranges[i+1])
                        {
                            indexHorizontal = i + 1;
                        }
                    }
                    if (v > ellipseV0)
                    {
                        indexVertical = 1;
                    }
                    int bin = indexVertical * nDivisions + indexHorizontal;
                    binMaps[map](u, v) = bin;

                    binArea[bin-1] ++;
                    ellipseArea ++;
                }
            }
        }

        //cout << "ellipse area: " << ellipseArea << endl;
        vector<int> fDist;
        for (int i = 0; i < nDivisions*2; i++)
        {
            double ratio = (double)binArea[i] / (double)ellipseArea * nFeatures;
            fDist.push_back((int)ratio);
            //cout << "bin " << i << " area: " << binArea[i] << endl;
            //cout << "bin " << i+1 << " features: " << featureDistribution[i] << endl;
        }

        featureDistributions.push_back(fDist);

        /*
        cv::Mat image = cv::Mat::zeros(height/2, width/2, CV_16UC1);
        for (int u = 0; u < width/2; u++)
        {
           for (int v = 0; v < height/2; v++)
           {
               double value = binMaps[map](u*2, v*2) * 65535 / (nDivisions * 2.0);
               image.at<short>(v,u) = (int)value;
           }
        }
        cv::imshow("map", image);
        cv::waitKey();
        cout << "Map computed" << endl;*/

    }
}
