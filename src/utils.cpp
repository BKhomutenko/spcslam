#include "utils.h"

void drawPoints(const vector<Eigen::Vector2d> & ptVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < ptVec1.size(); i++)
    {
        cv::circle(out, cv::Point(ptVec1[i](0), ptVec1[i](1)), 5, cv::Scalar(0, 255, 0));
    }
    for (int i = 0; i < ptVec2.size(); i++)
    {
        cv::circle(out, cv::Point(ptVec2[i](0), ptVec2[i](1)), 10, cv::Scalar(0, 0, 255));
    }
}

void drawPoints(const vector<Feature> & fVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < fVec1.size(); i++)
    {
        cv::circle(out, cv::Point(fVec1[i].pt(0), fVec1[i].pt(1)), 5, cv::Scalar(0, 255, 0));

    }
    for (int i = 0; i < ptVec2.size(); i++)
    {

        cv::circle(out, cv::Point(ptVec2[i](0), ptVec2[i](1)), 10, cv::Scalar(0, 0, 255));
    }
}

void drawPoints(const vector<Feature> & fVec1,
                const vector<Feature> & fVec2,
                cv::Mat & out, cv::Scalar color1, cv::Scalar color2)
{
    for (int i = 0; i < fVec1.size(); i++)
    {
        cv::circle(out, cv::Point(fVec1[i].pt(0), fVec1[i].pt(1)), 5, color1);

    }
    for (int i = 0; i < fVec2.size(); i++)
    {

        cv::circle(out, cv::Point(fVec2[i].pt(0), fVec2[i].pt(1)), 10, color2);
    }
}

void drawCrosses(const vector<Eigen::Vector2d> & ptVec, cv::Mat & out,
                 cv::Scalar color, double resizeRatio, double size, double thickness)
{
    double l = (double)size / 2.0;
    for (int i = 0; i < ptVec.size(); i++)
    {
        cv::line(out, cv::Point(ceil((ptVec[i](0) - l)*resizeRatio),
                                ceil((ptVec[i](1) - l)*resizeRatio)),
                      cv::Point(ceil((ptVec[i](0) + l)*resizeRatio),
                                ceil((ptVec[i](1) + l)*resizeRatio)), color, thickness);
        cv::line(out, cv::Point(ceil((ptVec[i](0) + l)*resizeRatio),
                                ceil((ptVec[i](1) - l)*resizeRatio)),
                      cv::Point(ceil((ptVec[i](0) - l)*resizeRatio),
                                ceil((ptVec[i](1) + l)*resizeRatio)), color, thickness);
        /*cv::line(out, cv::Point(ceil((ptVec[i](0) - l)*resizeRatio)-1,
                                ceil((ptVec[i](1) - l)*resizeRatio)),
                      cv::Point(ceil((ptVec[i](0) + l)*resizeRatio)-1,
                                ceil((ptVec[i](1) + l)*resizeRatio)), color, thickness);
        cv::line(out, cv::Point(ceil((ptVec[i](0) + l)*resizeRatio)+1,
                                ceil((ptVec[i](1) - l)*resizeRatio)),
                      cv::Point(ceil((ptVec[i](0) - l)*resizeRatio)+1,
                                ceil((ptVec[i](1) + l)*resizeRatio)), color, thickness);*/
    }
}

void drawCircles(const vector<Eigen::Vector2d> & ptVec, cv::Mat & out,
                 cv::Scalar color, double resizeRatio, double radius, double thickness)
{
    for (int i = 0; i < ptVec.size(); i++)
    {
        cv::circle(out, cv::Point(ptVec[i](0), ptVec[i](1))*resizeRatio, radius, color, thickness);
        //cv::circle(out, cv::Point(ptVec[i](0), ptVec[i](1))*resizeRatio, radius+1, color, thickness);
    }
}
