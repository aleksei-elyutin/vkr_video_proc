#include <iostream>
#include <iterator>
#include <string.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void drawKeypointCircle ( // Отрисовка особых точек цветными кругами
                         Mat& image,
                         vector<KeyPoint>& kps,
                         Scalar color
                        )
{
    for (size_t i = 0; i < kps.size(); i++ )
    {
        Point2f point = kps[i].pt;
        circle( image, point , 4, color , 1, 8, 0 );
    }

}

void drawMatchesLines ( // Отрисовка особых точек
                        Mat& image, //Изображение 1
                        vector<KeyPoint>& kps1, //Вектор особых точек изображения 1
                        vector<KeyPoint>& kps2, //Вектор особых точек изображения 2
                        Scalar kps1_color, //Цвет особых точек изображения 1
                        Scalar kps2_color, //Цвет особых точек изображения 2
                        Scalar lines_color //Цвет соединительных линий
                      )
{
    for (size_t i = 0; i < kps1.size(); i++ )
    {
        Point2f point1 = kps1[i].pt; Point2f point2 = kps2[i].pt;
        circle( image, point1 , 4, kps1_color , 1, 8, 0 );
        circle( image, point2 , 4, kps2_color , 1, 8, 0 );
        line(image,point1,point2,lines_color,1,8,0);
    }

}

bool selectDetector ( Ptr<Feature2D>& detector_obj,
                      string& detectorName
                    )
{
    if ((detectorName == "SURF")  |
        (detectorName == "KAZE")  |
        (detectorName == "AKAZE") |
        (detectorName == "BRISK") |
        (detectorName == "ORB") |
        (detectorName == "SIFT")   )
    {
        if (detectorName == "SURF") /** SURF creation **/
            {
                //detector_obj = SURF::create(); //defaults
                detector_obj = SURF::create(
                            700, // hessianThreshold - порог гессиана
                            3, // nOctaves - число октав
                            3, // nOctaveLayers - число уровней внутри каждой октавы
                            false, // использовать расширенный дескриптор
                            true); // использовать вычисление ориентации
               // cout << "SURF" << " - OK." << endl;
                return true;
             }
        if (detectorName == "KAZE")  /** KAZE creation **/
            {
                detector_obj = KAZE::create(); //defaults
                //cout << "KAZE" << " - OK." << endl;
                return true;
            }
        if (detectorName == "AKAZE")  /** AKAZE creation **/
            {
                detector_obj = AKAZE::create(); //defaults
                //cout << "AKAZE" << " - OK." << endl;
                return true;
            }
        if (detectorName == "BRISK")  /** BRISK creation **/
            {
                detector_obj = BRISK::create(); //defaults
                //cout << "BRISK" << " - OK." << endl;
                return true;
            }
        if (detectorName == "ORB")  /** FAST creation **/
            {
                detector_obj = ORB::create(); //defaults
                // cout << "ORB" << " - OK." << endl;
                return true;
            }
        if (detectorName == "SIFT")  /** FAST creation **/
            {
                detector_obj = SIFT::create(); //defaults
                //cout << "SIFT" << " - OK." << endl;
                return true;
            }
    }
    else
    {
        cout << "Incorrect detector name." << endl;
        return false;
    }
}

