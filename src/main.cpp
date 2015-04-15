#include "tests/cartography_tests.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv) {

    testCartography();
    
    //TRY CALIBRATION FUNCTIONS
    
    Size patternSize(5, 8);
    Mat frame = imread("/home/bogdan/projects/icars/calib_dataset/left_1426238811.pgm", 0);
    
    vector<Point2f> centers;
    bool patternfound = findChessboardCorners(frame, patternSize, centers);

    cout << patternfound << endl;
    
    drawChessboardCorners(frame, patternSize, Mat(centers), patternfound);
    
    for (auto & c : centers)
    {
        cout << c << endl;
    }
    
    imshow("corners", frame);
    waitKey();
}


