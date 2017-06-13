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
#include <opencv2/calib3d.hpp>

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

int addImFromMask (Mat& input_image1, Mat& input_image2, Mat& output_image, Mat& mask);

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
    namedWindow( "input stream", WINDOW_FULLSCREEN); //Создание окна
    //namedWindow( "warped mask", WINDOW_FULLSCREEN); //Создание окна

    string detectorName = parser.get<string>("detector");
    cout << "Try to init " << detectorName << "... "<< endl;

    Ptr<Feature2D> detector;

    if (selectDetector (detector, detectorName))
    {
        cout << detectorName << " - OK." << endl;
    }

    Mat current_frame, previous_frame, im_warped, im_fused, im_marked;
    vector<KeyPoint>  prev_kps, curr_kps, prev_kps_matched, curr_kps_matched;
    Mat prev_dscs, curr_dscs, mask;
    vector <Point2f> current_frame_pts, previous_frame_pts;


    FlannBasedMatcher matcher;
    //Ptr<DescriptorMatcher> matcher;
    //matcher = FlannBasedMatcher::create();
   // vector < vector< DMatch > > matches;
    vector <DMatch> matches;
    //vector <DMatch> single_matches;

    srcVideo.read(previous_frame);
    detector->detectAndCompute(previous_frame, Mat(), prev_kps, prev_dscs);
    im_fused = previous_frame.clone();

    while (srcVideo.read(current_frame))
    {
        detector->detectAndCompute(current_frame,Mat(),curr_kps, curr_dscs);
        detector->detectAndCompute(previous_frame, Mat(), prev_kps, prev_dscs);

        Mat immask = Mat::zeros(Size(current_frame.cols, current_frame.rows),current_frame.type());
        Mat warped_immask;

        matcher.match(prev_dscs, curr_dscs, matches);
        //>radiusMatch
        //cout << "Number of descriptors: " << curr_kps.size() << "   " << prev_kps.size() << endl;

        size_t size = matches.size();
        for( size_t i = 0; i < size; i++ )
        {
               // DMatch tempDM = matches[i].front();
               // single_matches.push_back(tempDM);
                if (matches[i].distance<100)
                {
                   // prev_kps_matched.push_back( prev_kps[ tempDM.queryIdx ] );
                   // curr_kps_matched.push_back( curr_kps[ tempDM.trainIdx ] );
                    previous_frame_pts.push_back( prev_kps[ matches[i].queryIdx ].pt );
                    current_frame_pts.push_back( curr_kps[ matches[i].trainIdx ].pt);
                }

        }
        Mat H = estimateAffinePartial2D(current_frame_pts,previous_frame_pts, mask, RANSAC);
        warpAffine(current_frame, im_warped, H, Size(current_frame.cols, current_frame.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
        warpAffine(immask, warped_immask , H , Size(current_frame.cols, current_frame.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
        addImFromMask (im_warped, previous_frame, im_fused, warped_immask);
        //cout << "Matches: " << curr_kps_matched.size() << endl;
        //frameQuery.push_back(im_result);
        //////// Добавить функцию сглаживания последовательности
        //addWeighted(im_fused , 0.9, warped_immask, 0.1, 0, im_marked);
       //drawMatchesLines (im_fused,curr_kps_matched,prev_kps_matched );

       // imshow( "output stream", im_fused);
        imshow( "input stream", current_frame);
        imshow( "output stream", im_fused);
        //imshow( "warped mask", warped_immask);

        /*if (counter++ >= 10)
        {
            previous_frame = current_frame.clone();
            counter=0;
        }*/
        previous_frame = im_fused.clone();
        prev_kps.clear(); //prev_kps = curr_kps; prev_dscs = curr_dscs.clone();
        curr_kps.clear();
        prev_kps_matched.clear();
        curr_kps_matched.clear();
        previous_frame_pts.clear();
        current_frame_pts.clear();

        //int key = cvWaitKey(33);
        //cout << "Pressed key: "<< key << endl;
        if ( cvWaitKey(1)  == 27 )  break; //27 for windows
    }

    return 0;
}
