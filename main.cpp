#include <iostream>
#include <iterator>
#include <deque>
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

    /** Functions protorypes */
void drawKeypointCircle ( // Отрисовка особых точек цветными кругами
                         Mat& image,
                         vector<KeyPoint>& kps,
                         Scalar color = Scalar(0, 0 , 255)
                        );
void drawMatchesLines ( // Отрисовка особых точек
                        Mat& image, //Изображение 1

                        vector<KeyPoint>& kps1, //Вектор особых точек изображения 1
                        vector<KeyPoint>& kps2, //Вектор особых точек изображения 2
                        Scalar kps1_color = Scalar(255, 0 , 0), //Цвет особых точек изображения 1
                        Scalar kps2_color = Scalar(255, 255 , 0), //Цвет особых точек изображения 2
                        Scalar lines_color = Scalar(0, 0 , 255) //Цвет соединительных линий
                        );
bool selectDetector ( Ptr<Feature2D>& detector_obj,
                      string& detectorName
                    );
    /*************************/
const string keys =
    "{help h usage ?  |      | print this message      } \
     {video           |      | path to video           } \
     {detector        |      | name of detector        }"
;

int main(int argc, char *argv[])
{
    cout << getBuildInformation() << endl;
    /** Parsing input **/  
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    } //Отобразить информационное сообщение

    VideoCapture srcVideo;
    if (parser.has("video"))
    {
        string inputStream_name = parser.get<string>("video");
        srcVideo = VideoCapture(inputStream_name);
    }
    else
    {
        srcVideo = VideoCapture(0);
    }

    if (!srcVideo.isOpened())
    {
        cout << "File invalid!" << endl;
        return -1;
    }



    /** Objects creation **/
    namedWindow( "output stream", WINDOW_FULLSCREEN); //Создание окна

    string detectorName = parser.get<string>("detector");
    cout << "Try to init " << detectorName << "... "<< endl;

    Ptr<Feature2D> detector;

    if (selectDetector (detector, detectorName))
    {
        cout << detectorName << " - OK." << endl;
    }

    Mat current_frame, previous_frame, im_result, im_fused;
    vector<KeyPoint>  prev_kps, curr_kps, prev_kps_matched, curr_kps_matched;
    Mat prev_dscs, curr_dscs;


    Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create("BruteForce");
    vector < vector< DMatch > > matches;
   // vector <DMatch> single_matches;

    srcVideo.read(previous_frame);
    detector->detectAndCompute(previous_frame, Mat(), prev_kps, prev_dscs);
    im_fused = previous_frame.clone();
    int counter = 0;
    while (srcVideo.read(current_frame))
    {
        detector->detectAndCompute(current_frame,Mat(),curr_kps, curr_dscs);
        //detector->detectAndCompute(previous_frame, Mat(), prev_kps, prev_dscs);

        matcher->radiusMatch(prev_dscs, curr_dscs, matches, 300);
        cout << "Number of descriptors: " << curr_kps.size() << "   " << prev_kps.size() << endl;

        for( size_t i = 0; i < matches.size(); i++ )
        {
            if (!matches[i].empty())
            {
                DMatch tempDM = matches[i].front();
               // single_matches.push_back(tempDM);
                if (tempDM.distance<250)
                {
                    prev_kps_matched.push_back( prev_kps[ tempDM.queryIdx ] );
                    curr_kps_matched.push_back( curr_kps[ tempDM.trainIdx ] );
                }
            }

        }


        cout << "Matches: " << curr_kps_matched.size() << endl;
        //frameQuery.push_back(im_result);
        //////// Добавить функцию сглаживания последовательности
        addWeighted(im_fused , 0.9, current_frame, 0.1, 0.0, im_fused);
        drawMatchesLines (im_fused,curr_kps_matched,prev_kps_matched );

        imshow( "output stream", im_fused);

        /*if (counter++ >= 10)
        {
            previous_frame = current_frame.clone();
            counter=0;
        }*/
        previous_frame = current_frame.clone();
        prev_kps.clear(); prev_kps = curr_kps; prev_dscs = curr_dscs.clone();
        curr_kps.clear();
        prev_kps_matched.clear();
        curr_kps_matched.clear();

        //int key = cvWaitKey(33);
        //cout << "Pressed key: "<< key << endl;
        if ( cvWaitKey(1)  == 27 )  break; //27 for windows
    }
/*
    vector<KeyPoint> im1_kps, im2_kps, im1_kps_matched, im2_kps_matched;
    Mat im1_dsc, im2_dsc;

    double timestamp1, timestamp2;  //Временные метки ***
    timestamp1   = (double)getTickCount();
    detector->detectAndCompute( img1, Mat(), im1_kps, im1_dsc);
    timestamp2 = (double)getTickCount();
    cout << "Time for detection and compution of frame 1: " << (timestamp2-timestamp1)/getTickFrequency() << endl;
    cout << "Number of features for image 1: " << im1_kps.size() << endl;
    timestamp1   = (double)getTickCount();
    detecto->detectAndCompute( img2, Mat(), im2_kps, im2_dsc);
    timestamp2 = (double)getTickCount();
    cout << "Time for detection and compution of frame 2: " << (timestamp2-timestamp1)/getTickFrequency() << endl;
    cout << "Number of features for image 2: " << im2_kps.size() << endl;

    Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create("BruteForce");
    vector < vector< DMatch > > matches;
    vector <DMatch> single_matches;
    matcher->radiusMatch(im1_dsc, im2_dsc, matches, 400);

    for( size_t i = 0; i < matches.size(); i++ )
    {
        if (!matches[i].empty())
        {
            DMatch tempDM = matches[i].front();
            single_matches.push_back(tempDM);
            im1_kps_matched.push_back( im1_kps[ tempDM.queryIdx ] );
            im2_kps_matched.push_back( im2_kps[ tempDM.trainIdx ] );
        }

    }
     cout << "Number of matched features: " << im1_kps_matched.size() << endl;


    Mat im_result;// = img1.clone();

    drawKeypointCircle(im_result, im1_kps_matched, Scalar(0, 255 , 0));
    drawKeypointCircle(im_result, im2_kps_matched, Scalar(255, 0 , 0));

    addWeighted(img1 , 0.5, img2, 0.5, 0.0, im_result);
    drawMatchesLines (im_result,im1_kps_matched,im2_kps_matched );
    imshow( "image1", im_result );
    cvWaitKey();
    //imshow( "image1", img2 );
    //cvWaitKey();
    */
    return 0;
}
