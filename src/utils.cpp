#include "utils.h"

void drawPoints(const vector<Eigen::Vector2d> & ptVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < ptVec1.size(); i++)
    {
        cv::circle(out, cv::Point(ptVec1[i](0), ptVec1[i](1)), 5, cv::Scalar(0, 255, 0));
        cv::circle(out, cv::Point(ptVec2[i](0), ptVec2[i](1)), 10, cv::Scalar(0, 0, 255));
    }

}

void drawPoints(const vector<Feature> & fVec1,
                const vector<Eigen::Vector2d> & ptVec2,
                cv::Mat & out)
{
    for (int i = 0; i < fVec1.size(); i++)
    {
      //  cv::circle(out, cv::Point(fVec1[i].pt(0), fVec1[i].pt(1)), 5, cv::Scalar(0, 255, 0));

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
