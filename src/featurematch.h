
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//#include "opencv2/core/eigen.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <utility> 

#include <Eigen/Dense>

#include "../five-point-nister/five-point.hpp"

#define MAXFEAT 2000
#define MINFEAT 400


using namespace Eigen;
using namespace cv;
using namespace std;
// Remove redundant matches
void removeMatches(vector<DMatch> &matches)
{
  cout << "matches size before: " << matches.size() << endl;

    for(int i = 0; i < matches.size(); i++){

      bool repeated = false;
      for(int j = i+1; j < matches.size(); j++){

        if(matches[i].trainIdx == matches[j].trainIdx){
          repeated = true;
          matches.erase(matches.begin() + j);
          j--;
        }
      }


      if(repeated)
        matches.erase(matches.begin() + i);
    }

    cout << "mathes size after: " << matches.size() << endl;

}


void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask, vector<Point2f> &pts_old, vector<Point2f> &pts_new)
    {
        int matchesCounter = 0;
        for (int i = 0; i < (int)matches.size(); i++)
        {
            if (mask.empty() || mask[i])
            {
                matchesCounter++;
                Point2f pt_new = query[matches[i].queryIdx].pt;
                Point2f pt_old = train[matches[i].trainIdx].pt;

                pts_new.push_back(pt_new);
                pts_old.push_back(pt_old);

                cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);

            }
        }
        //cout << "matchesCounter: " << matchesCounter << endl;
    }


//Takes a descriptor and turns it into an xy point
void keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(in[i].pt);
    }
}

//Takes an xy point and appends that to a keypoint structure
void points2keypoints(const vector<Point2f>& in, vector<KeyPoint>& out)
{
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out.push_back(KeyPoint(in[i], 1));
    }
}

//Uses computed homography H to warp original input points to new planar position
void warpKeypoints(const Mat& H, const vector<KeyPoint>& in, vector<KeyPoint>& out)
{
    vector<Point2f> pts;
    keypoints2points(in, pts);
    vector<Point2f> pts_w(pts.size());
    Mat m_pts_w(pts_w);
    perspectiveTransform(Mat(pts), m_pts_w, H);
    points2keypoints(pts_w, out);
}

//Converts matching indices to xy points
void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
    std::vector<Point2f>& pts_query)
{

    pts_train.clear();
    pts_query.clear();
    pts_train.reserve(matches.size());
    pts_query.reserve(matches.size());

    size_t i = 0;

    for (; i < matches.size(); i++)
    {

        const DMatch & dmatch = matches[i];

        pts_query.push_back(query[dmatch.queryIdx].pt);
        pts_train.push_back(train[dmatch.trainIdx].pt);
       // cout << dmatch.trainIdx << endl;

    }

}


void resetH(Mat&H)
{
    H = Mat::eye(3, 3, CV_32FC1);
}




bool DecomposeEtoRandT(
                                           Mat &E,
                                           Mat &R1,
                                           Mat &R2,
                                           Mat &t1,
                                           Mat &t2)
{

  SVD svd(E, SVD::MODIFY_A);

  //E is rank deficient , last singular value should be zero and the first and second 
  //should be same
  double singular_ratio = fabsf(svd.w.at<double>(0)/svd.w.at<double>(1));
  if( singular_ratio > 1.0)
    singular_ratio = 1/singular_ratio; //flip them to keep the ratio in[0,1]

  if (singular_ratio < 0.7)
  {
    cerr << "erroneous E computed, values are far apart";
  }

  Matx33d W(0, -1, 0,
              1, 0, 0,
              0 ,0, 1);
  Matx33d Wt(0, 1, 0,
              -1, 0, 0,
              0 ,0, 1);
  if (determinant(svd.u) < 0) svd.u = -svd.u; 
  if (determinant(svd.vt) < 0)  svd.vt = -svd.vt; 

  R1 = svd.u*Mat(W)*svd.vt;
  R2 = svd.u*Mat(Wt)*svd.vt;
  t1 = svd.u.col(2);
  t2 = -svd.u.col(2);

  return true;

}

std::pair <int,double> triangulateAndCheckReproj(const Mat& P, const Mat& P1, InputArray &prevFeatures,
                          InputArray &currFeatures, Mat &K, InputOutputArray &_mask ) {
  std::pair <int,double> result; // triangulation percentage and reproj error

    double dist = 100.0;



    //undistort
    Mat normalizedCurrPts,normalizedPrevPts;
    //undistortPoints(currFeatures, normalizedCurrPts, K, Mat());
    //undistortPoints(prevFeatures, normalizedPrevPts, K, Mat());

    prevFeatures.getMat().copyTo(normalizedPrevPts); 
    currFeatures.getMat().copyTo(normalizedCurrPts); 

    int npoints = normalizedPrevPts.checkVector(2);
  if (normalizedPrevPts.channels() > 1)
  {
    normalizedPrevPts = normalizedPrevPts.reshape(1, npoints); 
    normalizedCurrPts = normalizedCurrPts.reshape(1, npoints); 
  }

    normalizedPrevPts.col(0) = (normalizedPrevPts.col(0) - K.at<double>(0,2)) / K.at<double>(0,0); 
    normalizedCurrPts.col(0) = (normalizedCurrPts.col(0) - K.at<double>(0,2)) / K.at<double>(0,0); 
    normalizedPrevPts.col(1) = (normalizedPrevPts.col(1) - K.at<double>(1,2)) / K.at<double>(1,1); 
    normalizedCurrPts.col(1) = (normalizedCurrPts.col(1) - K.at<double>(1,2)) / K.at<double>(1,1); 
      

    normalizedPrevPts = normalizedPrevPts.t();
    normalizedCurrPts = normalizedCurrPts.t();

    //triangulation
    Mat pt_3d_h;//(4,currFeatures.size(),CV_32FC1);
    cv::triangulatePoints(P,P1,normalizedPrevPts,normalizedCurrPts,pt_3d_h);
    //cout << "after triangulatePoints" << endl;
    Mat mask1 = pt_3d_h.row(2).mul(pt_3d_h.row(3)) > 0; 
    //cout << "after mul" << endl;
    pt_3d_h.row(0) /= pt_3d_h.row(3); 
    pt_3d_h.row(1) /= pt_3d_h.row(3); 
    pt_3d_h.row(2) /= pt_3d_h.row(3); 
    pt_3d_h.row(3) /= pt_3d_h.row(3); 
    mask1 = (pt_3d_h.row(2) < dist) & mask1;
    
    //cout << "before P1*pt_3d_h" << endl;
    pt_3d_h.convertTo(pt_3d_h, P1.type());
    cout << P1.type() << " , " << pt_3d_h.type() << endl;
    pt_3d_h = P1 * pt_3d_h;
    //pt_3d_h = P1.mul(pt_3d_h); 
    //cout << "after P1*pt_3d_h" << endl;
    mask1 = (pt_3d_h.row(2) > 0) & mask1; 
    mask1 = (pt_3d_h.row(2) < dist) & mask1; 

    cout << "after the triangulation part" << endl;

    // If _mask is given, then use it to filter outliers. 
    if (!_mask.empty())
    {
    Mat mask = _mask.getMat();         
        CV_Assert(mask.size() == mask1.size()); 
    bitwise_and(mask, mask1, mask1); 
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U); 
    }


    /*Mat pt_3d; convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1),pt_3d);
    //    cout << pt_3d.size() << endl;
    //    cout << pt_3d.rowRange(0,10) << endl;
    vector<uchar> status(pt_3d.rows,0);
    for (int i=0; i<pt_3d.rows; i++) {
        status[i] = (pt_3d.at<Point3f>(i).z > 0) ? 1 : 0;
        //cout << pt_3d.at<Point3f>(i) << endl;
    }*/
    int count = countNonZero(mask1);

    /*double percentage = ((double)count / (double)pt_3d_h.cols);
    cout << count << "/" << pt_3d_h.cols << " = " << percentage*100.0 << "% are in front of camera";

    //calculate reprojection
    cv::Mat_<double> R = P(cv::Rect(0,0,3,3));
    Vec3d rvec(0,0,0); //Rodrigues(R ,rvec);
    Vec3d tvec(0,0,0); // = P.col(3);
    vector<Point2f> reprojected_pts;
    projectPoints(pt_3d_h(Rect(0,0,3,pt_3d_h.cols)),rvec,tvec,K,Mat(),reprojected_pts);
//    cout << Mat(reprojected_pt_set1).rowRange(0,10) << endl;
    double reprojErr = cv::norm(reprojected_pts,prevFeatures,NORM_L2)/(double)prevFeatures.size();
    //cout << "reprojection Error " << reprojErr;*/

    return std::make_pair (count,0.2); 
}

void relativeTransformation(Mat &E, Mat &R, Mat &t, Mat &K, vector<Point2f> &prevFeatures,
                            vector<Point2f> &currFeatures, vector<uchar> &mask)
{
      std::pair <int,double> result1;
      std::pair <int,double> result2;
      std::pair <int,double> result3;
      std::pair <int,double> result4;

        if(fabsf(determinant(E)) > 1e-07) {
            cout << "det(E) != 0 : " << determinant(E);
            cout << "there is error computing E" << endl;
        }

        Mat_<double> R1(3,3);
        Mat_<double> R2(3,3);
        Mat_<double> t1(1,3);
        Mat_<double> t2(1,3);
        if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) cerr << "erroneous E" << endl;

        if(determinant(R1)+1.0 < 1e-09) {
            //according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            //cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign";
            //E = -E;
            if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) cerr << "erroneous E" << endl;
        }
        if(fabsf(determinant(R1))-1.0 > 1e-07) {
            cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
      
        }

        Mat P = Mat::eye(3,4,CV_64FC1);

        
        Mat_<double> P1 = (Mat_<double>(3,4) <<
                           R1(0,0),   R1(0,1),    R1(0,2),    t1(0),
                           R1(1,0),   R1(1,1),    R1(1,2),    t1(1),
                           R1(2,0),   R1(2,1),    R1(2,2),    t1(2));
        //cout << "P1\n" << Mat(P1) << endl;

        result1 = triangulateAndCheckReproj( P,  P1, prevFeatures,
                          currFeatures, K, mask ) ;

        //cout << "status1: " << result1.first << endl;

        Mat_<double> P2 = (Mat_<double>(3,4) <<
                           R2(0,0),   R2(0,1),    R2(0,2),    t1(0),
                           R2(1,0),   R2(1,1),    R2(1,2),    t1(1),
                           R2(2,0),   R2(2,1),    R2(2,2),    t1(2));
        //cout << "P2\n" << Mat(P2) << endl;

        result2 = triangulateAndCheckReproj(P,  P2, prevFeatures,
                          currFeatures, K, mask);

        //cout << "status2: " << result2.first << endl;

         Mat_<double> P3 = (Mat_<double>(3,4) <<
              R1(0,0),   R1(0,1),    R1(0,2),    t2(0),
              R1(1,0),   R1(1,1),    R1(1,2),    t2(1),
              R1(2,0),   R1(2,1),    R1(2,2),    t2(2));
        //cout << "P3\n" << Mat(P3) << endl;

        result3 = triangulateAndCheckReproj(P,  P3, prevFeatures,
                          currFeatures, K , mask);
        //cout << "status3: " << result3.first << endl;
        
        Mat_<double> P4 = (Mat_<double>(3,4) <<
                           R2(0,0),   R2(0,1),    R2(0,2),    t2(0),
                           R2(1,0),   R2(1,1),    R2(1,2),    t2(1),
                           R2(2,0),   R2(2,1),    R2(2,2),    t2(2));
        //cout << "P4\n" << Mat(P4) << endl;

        result4 = triangulateAndCheckReproj(P,  P4, prevFeatures,
                          currFeatures, K, mask);
        //cout << "status4: " << result4.first << endl;

        

        if (result1.first >= result2.first && result1.first >= result3.first && result1.first >= result4.first)
        {
          R = R1; t = t1; 
          cout << "status1: " << result1.first << endl;
          cout << "error: " << result1.second << endl;

        }
        else if (result2.first >= result1.first && result2.first >= result3.first && result2.first >= result4.first)
        {
          R = R1; t = t2;
          cout << "status2: " << result2.first << endl;  
          cout << "error: " << result2.second << endl;

        }
        else if (result3.first >= result1.first && result3.first >= result2.first && result3.first >= result4.first)
        {
          R = R2; t = t2; 
          cout << "status3: " << result3.first << endl;
          cout << "error: " << result3.second << endl;

        }
        else  
        {
          R = R2; t = t1;
          cout << "status4: " << result4.first << endl;
          cout << "error: " << result4.second << endl;
  
        }

        
      
    return ;
}

//Note Prev => Train, Curr => Query
int featureMatching(Mat &img, Mat &SIFT_outputImg, Mat &K, Mat &R, Mat &t, Mat &tvec)
{
  int features_count = 0;
  //cout << "entered featureMatching" << endl;
  cv::Mat imGray;
  if(img.channels() == 3){
     cvtColor(img, imGray, CV_RGB2GRAY);
  }
     
  else{
     img.copyTo(imGray);
  }
     

    // SIFT...
  img.copyTo(SIFT_outputImg);

  static const cv::Ptr<cv::FeatureDetector> SIFT_detector(new cv::SiftFeatureDetector());
  static const cv::Ptr<cv::DescriptorExtractor> SIFT_descriptor(new cv::SiftDescriptorExtractor());
  static const cv::Ptr<cv::DescriptorMatcher> SIFT_matcher(DescriptorMatcher::create("BruteForce"));


  //static const cv::Ptr<cv::FeatureDetector> SIFT_detector(new cv::GridAdaptedFeatureDetector(new FastFeatureDetector(10, true), MAXFEAT, 4, 4));
  //static const cv::Ptr<cv::DescriptorExtractor> SIFT_descriptor(new cv::BriefDescriptorExtractor(32));
  //static const cv::Ptr<cv::DescriptorMatcher> SIFT_matcher(DescriptorMatcher::create("BruteForce-Hamming"));
  static Mat SIFT_H_prev;

  vector<Point2f> currFeatures, prevFeatures;
  Mat E;
  vector<uchar> mask;
  static vector<KeyPoint> SIFT_train_kpts; 
  vector<KeyPoint> SIFT_query_kpts;
  static Mat SIFT_train_desc; 
  Mat SIFT_query_desc;
  std::vector<Point2f> SIFT_train_pts,  SIFT_query_pts;
  std::vector<cv::DMatch> SIFT_matches;
  std::vector<unsigned char> SIFT_match_mask;

  SIFT_detector->detect(imGray, SIFT_query_kpts);
  SIFT_descriptor->compute(imGray, SIFT_query_kpts, SIFT_query_desc);

  //cout << "detect and compute done " << endl;

  if(SIFT_H_prev.empty()){
    SIFT_H_prev = Mat::eye(3,3,CV_32FC1);
  }


  if(!SIFT_train_kpts.empty())
  {

    std::vector<cv::KeyPoint> test_kpts;
    warpKeypoints(SIFT_H_prev.inv(), SIFT_query_kpts, test_kpts);
    cv::Mat SIFT_mask = windowedMatchingMask(test_kpts, SIFT_train_kpts, 25, 25);
    SIFT_matcher->match(SIFT_query_desc, SIFT_train_desc, SIFT_matches, SIFT_mask);
    //removeMatches(SIFT_matches);
    matches2points(SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_train_pts, SIFT_query_pts);
    //cout << "matches done " << endl;
        
    if(SIFT_matches.size() >= 5)
    {
      cv::Mat H = findHomography(SIFT_train_pts, SIFT_query_pts, CV_RANSAC, 4.0, SIFT_match_mask);
      /*int j= 0;
      for(size_t i= 0; i< SIFT_matches.size(); ++i)
      {
        if(int(SIFT_match_mask[i]) == 0)
        {
          SIFT_matches.erase(SIFT_matches.begin() + j);
          SIFT_train_pts.erase(SIFT_train_pts.begin() + j);
          SIFT_query_pts.erase(SIFT_query_pts.begin() + j);
          --j;

        }
        ++j;

      }*/

      if(countNonZero(Mat(SIFT_match_mask)) >= 15)
      {
         SIFT_H_prev = H;
  
        //cout << "temporal matches count: " << countNonZero(Mat(SIFT_match_mask)) << endl;
      }
      else{

        SIFT_H_prev = Mat::eye(3,3,CV_32FC1);
      }

      //cout << "before drawMatchesRelative" << endl;

      
      drawMatchesRelative(SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_outputImg, SIFT_match_mask, prevFeatures, currFeatures );
      E = findEssentialMat(currFeatures, prevFeatures, K.at<double>(0,0), Point2d(K.at<double>(0,2),K.at<double>(1,2)) , RANSAC, 0.999, 1.0, mask);
      //cout << "masksize: " << mask.size() << "countNonZero: " << countNonZero(mask) << endl;
      Mat pt_3d;
      features_count = recoverPose(E, currFeatures, prevFeatures, R, t, pt_3d, K.at<double>(0,0), Point2d(K.at<double>(0,2),K.at<double>(1,2)), mask);
      //pt_3d = pt_3d.t();
      //cout << "pt_3d.channels: " <<  pt_3d.channels() << endl;
      //Mat rvec; // tvec;
      //tvec = t;
      //Rodrigues (R,rvec);
     // solvePnPRansac(pt_3d, currFeatures, K, Mat(), rvec, tvec, false, CV_EPNP);
      //cout << "tvec: " << tvec << endl;
      //Rodrigues(rvec, R);
      //cout << "pt_3d shape: " << pt_3d.rows << " , " << pt_3d.cols << endl;
      //for (int i=0; i< int(pt_3d.cols) ; ++i)
        //cout << pt_3d.at<double>(2,i) << endl;

      //cout << "R1: " << R << endl;
      //cout << "t1: " << t << endl;

      //relativeTransformation(E, R, t, K, prevFeatures, currFeatures, mask);
      //cout << "R2: " << R << endl;
      //cout << "t2: " << t << endl;
    }
  }
  else
  { 
    SIFT_H_prev = Mat::eye(3,3,CV_32FC1);
  }



  //if (prevFeatures.size() <= MINFEAT)
  //{
   // cout << "currFeatures size: " << currFeatures.size() << endl;
   // cout << " redetecting features prevFeatures.size():  " << prevFeatures.size()<< endl;
    SIFT_train_kpts = SIFT_query_kpts;
    SIFT_query_desc.copyTo(SIFT_train_desc); 

  //}
  
  //cout << "prev items copied" << endl;
  
  if(true)
  {
    cout << ", SIFT matches: " << SIFT_matches.size();
  }

  return features_count;

}

