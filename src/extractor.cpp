
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Eigen>

#include "extractor.h"


using namespace std;
using namespace cv;
typedef Mat_<float> fMat;
typedef Mat_<uchar> uMat;
void FeatureExtractor::compute(const fMat & src, vector<Feature> & featureVec)
{

    findFeatures(src);

    finalizeFeatures(src, featureVec);
}

void FeatureExtractor::findFeatures(const fMat & src)
{
//    computeNormImage(src);
    computeGradients(src);
    computeResponse(1.5f);
    
    findMaxima();
}

void FeatureExtractor::finalizeFeatures(const fMat & src, vector<Feature> & featureVec)
{
    int finalFeatNum = min(numFeatures, int(maxVec.size()));
    partial_sort(maxVec.begin(), maxVec.begin() + finalFeatNum, maxVec.end(), pairCompare<Vector2d>);
    cout << finalFeatNum << endl;
    featureVec.reserve(finalFeatNum);
    for (auto fIter = maxVec.begin(); fIter != maxVec.begin() + finalFeatNum; ++fIter)
    {
        Vector2d pt = fIter->second;
        Descriptor d;
        computeDescriptor(src, pt, d);
        featureVec.push_back(Feature(pt, d, 1, 1));
    }
}

template<typename T>
T sign(const T & x)
{
    return T((x > 0) - (x < 0));
}


void FeatureExtractor::computeDescriptor(const fMat & src, Vector2d pt, Descriptor & d)
{
    int patchSize = 24;
    double alpha = 5, beta = 1;
    fMat patch, patchx, patchy, patchl;
    fMat patchr(Size(patchSize, patchSize));
    Point cvpt(pt[0], pt[1]);
    getRectSubPix(src, Size(patchSize, patchSize), cvpt, patch, CV_32F);
    
    patch = patch/alpha + beta;
    log(patch, patchl);

    
    getRectSubPix(gradx, Size(patchSize, patchSize), cvpt, patchx, CV_32F);
    GaussianBlur(patchx, patchx,  Size(0, 0), 1.2);
    patchx /= patch;
    
    
    getRectSubPix(grady, Size(patchSize, patchSize), cvpt, patchy, CV_32F);
    
    patchy /= patch;
    
    double b = (patchSize - 1)/2.; // the center of the patch
    for (unsigned int i = 0; i < patchSize; i++)
    {
        for (unsigned int j = 0; j < patchSize; j++)
        {
            patchr(i, j) = patchy(i,j) * (i - b) + patchx(i,j) * (j - b);
        }
    }
    
    GaussianBlur(patchl, patchl,  Size(0, 0), 1.5);
    resize(patchl, patchl, descSize);
    copy(patchl.begin(), patchl.end(), d.data());
    
    GaussianBlur(patchx, patchx,  Size(0, 0), 1.5);
    resize(patchx, patchx, descSize);
    copy(patchx.begin(), patchx.end(), d.data() + N*N);
    
    GaussianBlur(patchy, patchy,  Size(0, 0), 1.5);
    resize(patchy, patchy, descSize);
    copy(patchy.begin(), patchy.end(), d.data() + 2*N*N);
    
    GaussianBlur(patchr, patchr,  Size(0, 0), 1.5);
    resize(patchr, patchr, descSize);
    copy(patchr.begin(), patchr.end(), d.data() + 3*N*N);
//    double dmean = d.mean();
//    for (unsigned int i = 0; i < d.innerSize(); i++)
//    {
//        d[i] -= dmean;
//    }
//    d /= d.norm();
    /*
    int u = pt[0];
    int v = pt[1];
    d.setZero();
    float w[6] = {0, 0.5, 0, 1, 0, 0.5};
    for (int i : {1, 3, 5})
    {
        for (int j : {1, 3, 5})
        {
            float wtotal = w[i]*w[j];
            d[0] += gradx(v - i, u - j)*wtotal;
            d[1] += grady(v - i, u - j)*wtotal;
            d[2] += gradx(v - i, u + j)*wtotal;
            d[3] += grady(v - i, u + j)*wtotal;
            d[4] += gradx(v + i, u - j)*wtotal;
            d[5] += grady(v + i, u - j)*wtotal;
            d[6] += gradx(v + i, u + j)*wtotal;
            d[7] += grady(v + i, u + j)*wtotal;
        }
    }
    for (unsigned int i = 0; i < 8; i++)
    {
        d[i] = sign(d[i]) * log(abs(d[i]) + 1);
    }
    */
}

void testResponse(const fMat & src)
{
    fMat foo;
    GaussianBlur(src, foo, Size(0, 0), 1.5);
    foo = src / (foo + 1);
    imshow("foo", foo/2);
    waitKey();
}

void FeatureExtractor::computeNormImage(const fMat & src)
{
    fMat foo;
    GaussianBlur(src, foo, Size(0, 0), 3.5);
    normImg = src - foo;
    foo = normImg.mul(normImg);
    GaussianBlur(foo, foo, Size(0, 0), 3.5);
    sqrt(foo, foo);
    normImg /= (foo+1);
    imshow("111", normImg*0.2 + 0.5);
        
}

void FeatureExtractor::computeGradients(const fMat & src)
{
//    testResponse(src);
    /*Sobel(normImg, Ixx, CV_32F, 2, 0, 3, 1/8.);
    Sobel(normImg, Ixy, CV_32F, 1, 1, 3, 1/8.);
    Sobel(normImg, Iyy, CV_32F, 0, 2, 3, 1/8.);*/
    uMat uSrc;
    fMat imgEqual;
    src.convertTo(uSrc, CV_8U);
    equalizeHist(uSrc, uSrc);
    uSrc.convertTo(imgEqual, CV_32F);
    normImg = 0.5*src + 0.5*imgEqual;
    imshow("equalized", normImg/255);
    Sobel(normImg, gradx, CV_32F, 1, 0, 3, 1/8.);
    Sobel(normImg, grady, CV_32F, 0, 1, 3, 1/8.);
    Ixx = gradx.mul(gradx);
    Ixy = grady.mul(gradx);
    Iyy = grady.mul(grady);
   /* GaussianBlur(gradx, gradx, Size(0, 0), 1.5);
    GaussianBlur(grady, grady, Size(0, 0), 1.5);*/
}

void FeatureExtractor::computeResponse(float scale)
{
    fMat Jx, Jy, Jzx, Jzy, Jz, Jxx, Jxy, Jyy;
    /*GaussianBlur(normImg.mul(gradx), Jx, Size(0, 0), scale);
    GaussianBlur(normImg.mul(grady), Jy, Size(0, 0), scale);
    GaussianBlur(gradx.mul(gradx), Ixx, Size(0, 0), scale);
    GaussianBlur(gradx.mul(grady), Ixy, Size(0, 0), scale);
    GaussianBlur(grady.mul(grady), Iyy, Size(0, 0), scale);

    Jxx = 2*Jx.mul(Jx) / normImg/normImg/normImg/normImg + Ixx / normImg / normImg;
    Jxy = 2*Jx.mul(Jy) / normImg/normImg/normImg/normImg + Ixy / normImg / normImg;
    Jyy = 2*Jy.mul(Jy) / normImg/normImg/normImg/normImg + Iyy / normImg / normImg;*/
    GaussianBlur(Ixx, Jxx, Size(0, 0), scale);
    GaussianBlur(Ixy, Jxy, Size(0, 0), scale);
    GaussianBlur(Iyy, Jyy, Size(0, 0), scale);

    fMat trace = (Jxx + Jyy + 1);
    response = std::sqrt(scale) * (Jxx.mul(Jyy) - Jxy.mul(Jxy)) / trace;
    imshow("response", response/10);
    waitKey();
}

/*
//TODO extend scale to vector<float> scales
void Extractor::findFeatures(const fMat & src, std::vector<KeyPoint> & points, int camId,
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

        fMat resp;
        computeResponse(src, resp, scale);

        vector<Point2f> maxPoints;
        cout << 222 << endl;
        findMax(resp, maxPoints, 0.9, camId);


        for (int i = 0; i < maxPoints.size(); i++)
        {
            points.push_back(KeyPoint(maxPoints[i].x, maxPoints[i].y, (scale*0.5f+4)));
        }
    }
}
*/

void FeatureExtractor::findMaxima()
{
    int cols = response.cols;
    float * srcPtr = (float *)(response.data);
    int neiu[] = {-1, 1, -1, 1, 0, 0, 1, -1};
    int neiv[] = {-1, 1, 1, -1, 1, -1, 0, 0};
    int margin = 50;
    maxVec.clear();
    for (int v = margin; v < response.rows - margin-300; v++)
    {
        for (int u = margin; u < response.cols - margin; u++)
        {
            bool isMax = true;
            float srcVal = srcPtr[v*cols + u];
            if (srcVal < thresh) continue;
            for (int i = 0; i < 8; i++)
            {
                if(srcVal <= srcPtr[(v + neiv[i])*cols + u + neiu[i]])
                {
                    isMax = false;
                    break;
                }
            }
            if (isMax == false)
            {
                continue;
            }
            maxVec.push_back(make_pair(-srcVal, Vector2d(u, v)));
        }
    }
}
/*
void Extractor::extractDescriptor(const fMat & src, Point2f pt, int patchSize, Size descSize, fMat & dst)
{
    fMat patch;
    getRectSubPix(src, Size(patchSize, patchSize), pt, patch, CV_32F);
    resize(patch, patch, descSize);
    patch = patch.mul(kernel);
    patch = patch.reshape(0, 1);
    fMat mean, stddev;

    patch.copyTo(dst);
}

void Extractor::extractFeatures(const fMat & src, vector<KeyPoint> points, fMat & descriptors)
{
    kernel = getGaussianKernel(descWidth, descWidth/2, CV_32F);
    kernel = kernel * kernel.t();
    descriptors = fMat(points.size(), descWidth*descWidth, CV_32F);
    Size descSize = Size(descWidth, descWidth);

    for (int i = 0; i < points.size(); i++)
    {
        fMat row = descriptors.row(i);
        int patchSize = points[i].size * 2;

        extractDescriptor(src, points[i].pt, points[i].size, descSize, row);

    }
}

void Extractor::computeMaps()
{
    int width = 1296;
    int height = 966;

    for (int mapIdx = 0; mapIdx < 2; mapIdx++)
    {
        Eigen::MatrixXi m;
        m.resize(width, height);
        m.setZero();
        binMaps.push_back(m);

        vector<double> uRanges(nDivHorizontal + 1);
        vector<double> vRanges(nDivVertical + 1);
        vector<int> subArea(nDivHorizontal * nDivVertical, 0);
        int maskArea = 0;
        double binWidth = (uBounds[mapIdx][1] - uBounds[mapIdx][0]) / (double)nDivHorizontal;
        double binHeight = (vBounds[1] - vBounds[0]) / (double) nDivVertical;

        for (int i = 0; i < nDivVertical + 1; i++)
        {
            vRanges[i] = vBounds[0] + binHeight * i;
            //cout << "vRange: " << vRanges[i] << endl;
        }
        for (int i = 0; i < nDivHorizontal + 1; i++)
        {
            uRanges[i] = uBounds[mapIdx][0] + binWidth * i;
            //cout << "uRange: " << uRanges[i] << endl;
        }

        for (int u = 0; u < width; u++)
        {
            for (int v = 0; v < height; v++)
            {
                int indexU = 0;
                int indexV = 0;


                if ( (u-circle1[mapIdx][0]) * (u-circle1[mapIdx][0]) +
                     (v-circle1[mapIdx][1]) * (v-circle1[mapIdx][1])
                     <= circle1[mapIdx][2] * circle1[mapIdx][2] and
                     (u-circle2[mapIdx][0]) * (u-circle2[mapIdx][0]) +
                     (v-circle2[mapIdx][1]) * (v-circle2[mapIdx][1])
                     <= circle2[mapIdx][2] * circle2[mapIdx][2] and
                     v >= vBounds[0] and v <= vBounds[1] )
                {
                    for (int i = 0; i < nDivVertical; i++)
                    {
                        if (v > vRanges[i] and v <= vRanges[i+1])
                        {
                            indexV = i;
                            break;
                        }
                    }
                    for (int i = 0; i < nDivHorizontal; i++)
                    {
                        if (u > uRanges[i] and u <= uRanges[i+1])
                        {
                            indexU = i + 1;
                            break;
                        }
                    }

                    int bin = indexV * nDivHorizontal + indexU;
                    binMaps[mapIdx](u, v) = bin;
                    subArea[bin]++;
                    maskArea ++;
                }
            }
        }

        //cout << "ellipse area: " << ellipseArea << endl;
        vector<int> fDist;
        for (int i = 0; i < nDivVertical * nDivHorizontal; i++)
        {
            double ratio = (double)subArea[i] / (double)maskArea * nFeatures;
            if (i % 2 == 0)
            {
                fDist.push_back(floor(ratio));
            }
            else
            {
                fDist.push_back(ceil(ratio));
            }
            //cout << "bin " << i+1 << " area: " << subArea[i] << endl;
            //cout << "bin " << i+1 << " features: " << fDist[i] << endl;
        }

        featureDistributions.push_back(fDist);

        if (mapIdx == 0)
        {
            fMat image = fMat::zeros(height/2, width/2);
            for (int u = 0; u < width/2; u++)
            {
                for (int v = 0; v < height/2; v++)
                {
                    double value = (double)binMaps[mapIdx](u*2, v*2) * 65535 / (nDivVertical * nDivHorizontal);
                    image.at<short>(v,u) = (int)value;
                }
            }
            //imshow("map", image);
            //cout << "Map computed" << endl;
            //waitKey(100);
        }
    }
}

*/

/*void Extractor::computeBinMaps()
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
        fMat image = fMat::zeros(height/2, width/2, CV_16UC1);
        for (int u = 0; u < width/2; u++)
        {
           for (int v = 0; v < height/2; v++)
           {
               double value = binMaps[map](u*2, v*2) * 65535 / (nDivisions * 2.0);
               image.at<short>(v,u) = (int)value;
           }
        }
        imshow("map", image);
        waitKey();
        cout << "Map computed" << endl;

    }
}*/
