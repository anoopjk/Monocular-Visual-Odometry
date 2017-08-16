

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

int main( int argc, char** argv )
{
  Mat currImage, prevImage;
  int framenum = 0;
  ofstream output;
  output.open ("voOutput.txt");

  double scale = 1.00;

  String folderpath = "../IMG/*.png" ;
  vector<String> filenames;
  cv::glob(folderpath, filenames);


  Mat K(3, 3, CV_64F);
  K.at<double>(0,0) = 1075.62 ;  K.at<double>(0,1) = 0.; K.at<double>(0,2) = 621.462 ;
  K.at<double>(1,0) = 0.; K.at<double>(1,1) = 1075.62 ; K.at<double>(1,2) = 358.773;
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

  //cout << "before while loop" << endl;
  cout << "no.of images: " << filenames.size() << endl;
  for(int framenum = 0; framenum < int(filenames.size()); ++framenum)
    {
        cout << "Image#: " <<  framenum << endl;
        if (framenum >=0){

            
            currImage = imread(filenames[framenum]);
            cout << filenames[framenum] << endl;
            features_count = featureMatching(currImage, prevImage, K, R, t, tvec)

            if ( (!t.empty() && features_count > 5) ) 
            {
                //cout << "scale*t :" << scale*t << endl;   
                 t_f = t_f + scale*(R_f*t);
                 R_f = R*R_f;
            }
            
            else {
            }
            output << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2)  << endl;
             
            output.flush();
            
            imshow("image", prevImage);
            waitKey(1);

        }

    }                                  
   

 return 0;
}

  

