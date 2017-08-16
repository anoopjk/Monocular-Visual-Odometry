/*main file for processing the data and running the EKF*/


// #include "features.h"
// #include "EKF.h"


#define MAX_FRAME 3000
#define MIN_NUM_FEAT 300

#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <limits>

//#include "features.h"
#include "featurematch.h"
#include "../five-point-nister/five-point.hpp"
//#include "MVO.h"
#include "../fast-cpp-csv-parser/csv.h"
#include <math.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace Eigen;
using namespace cv;
//The following function is based on the same from https://github.com/avisingh599/mono-vo
double getAbsoluteScale(int frame_id)  {
  
  string line;
  int i = 0;
  ifstream myfile ("/home/anoop/Documents/robotics/EKF_mono_slam/KITTI01/dataset/poses/02.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}
int main( int argc, char** argv )
{
  Mat currImage, prevImage;
  int framenum = 0;
  ofstream output;
  output.open ("voOutput.txt");

  double scale = 1.00;

  String folderpath = "/home/anoop/Documents/yembo/processedIMG/*.png" ; //"/home/anoop/Documents/robotics/EKF_mono_slam/KITTI02/image_2/*.png";
  vector<String> filenames;
  cv::glob(folderpath, filenames);


  Mat K(3, 3, CV_64F);
  K.at<double>(0,0) = 1075.62   /*7.188560000000e+02*/ ;  K.at<double>(0,1) = 0.; K.at<double>(0,2) = 621.462 /*6.071928000000e+02 */;
  K.at<double>(1,0) = 0.; K.at<double>(1,1) = 1075.62  /*7.188560000000e+02*/ ; K.at<double>(1,2) = 358.773 /*1.852157000000e+02 */;
  K.at<double>(2,0) = 0.; K.at<double>(2,1) = 0.; K.at<double>(2,2) = 1.;

  Mat D(5, 1, CV_64F);
  D.at<double>(0,0) = 0.0;
  D.at<double>(0,1) = 0.0;
  D.at<double>(0,2) = 0.0;
  D.at<double>(0,3) = 0.0;
  D.at<double>(0,4) = 0.0;
  
  Mat R_f = Mat::eye(3,3,CV_64F);
  Mat t_f = Mat::zeros(3,1,CV_64F);
  Mat R, t,tvec, E, mask;
  int features_count = 0;

  cout << "before while loop" << endl;
  cout << "no.of images: " << filenames.size() << endl;
    while( true /*framenum <=3000*/)
    {
        cout << "Image#: " <<  framenum << endl;
        if (framenum >=0){

            
            currImage = imread(filenames[framenum]);
            cout << filenames[framenum] << endl;
            features_count = featureMatching(currImage, prevImage, K, R, t, tvec);

            //scale
            //scale = getAbsoluteScale(framenum ); 
            //cout << "Scale is " << scale << endl;

            if ( (!t.empty() && features_count > 5) /*&& (scale>0.1) &&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))*/) 
            {
                //cout << "scale*t :" << scale*t << endl;   
                 t_f = t_f + scale*(R_f*t);
                 R_f = R*R_f;
            }
            
            else {
           //cout << "scale below 0.1, or incorrect translation" << endl;
            }
            output << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2)  << endl;
             
            output.flush();
            
            imshow("image", prevImage);
            waitKey(1);

        }
    
    framenum = framenum+1; // increment the image counter

    }                                  
   

 return 0;
}

  

