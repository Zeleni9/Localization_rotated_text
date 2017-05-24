#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include "opencv2/photo.hpp"

using namespace cv;
using namespace std;

// Global variables
Mat image; 
double pi = 4*atan(1.0);   // Maximum precision available on any architecture for Pi number

// Finding threshold from percentage of histogram   
int findThreshold(int histogram[], int total, double percentage) {

	int curr_sum = 0;
	int threshold = 0; 
	int total_pixels = total*percentage; 
	for(int i=0; i<256; ++i) {
			curr_sum += histogram[i]; 
		if(curr_sum > total_pixels) {
			return i; 
		}
	}
}

// Filter Gxx(theta) gives maximal/minimal response to a rotation angle
double calculateGTheta(double Gxx,double Gyy, double Gxy, double theta) { 
	return Gxx * (cos(theta)*cos(theta)) + Gyy*(sin(theta)*sin(theta)) - 2*Gxy*(cos(theta)*sin(theta)); 
}


void steerable_filter(Mat &image, Mat &filterImage,  double sigma) {

	int filter_size = (int)(2*sigma + 0.5); 		// Size of filter 
	int region_size = 2*filter_size + 1; 			// Size of filter matrix ( region_size*region_size)

	// Initializations of filter's matrix
	std::vector< std::vector<double> > matrixGxx(region_size, std::vector<double>(region_size, 0)); 
	std::vector< std::vector<double> > matrixGyy(region_size, std::vector<double>(region_size, 0)); 
	std::vector< std::vector<double> > matrixGxy(region_size, std::vector<double>(region_size, 0)); 

	for(int i=0; i<region_size; i++) {
		for (int j=0; j<region_size; j++) {		
			double x = i - filter_size;  // vrijednost filta Gxx/Gyy/Gxy se računa sa x -> udaljenost po x-osi od središnjeg piksela do piksela za koji se računa vrijednost filtra
			double y = j - filter_size;  //  y -> isto po y-osi 

			// Calculating Gxx,Gyy,Gxy filters 
			matrixGxx[i][j] = (x*x - sigma*sigma)/(2*pi*pow(sigma,6)) *exp(-(x*x+y*y)/(2*sigma*sigma)); 
			matrixGyy[i][j] = (y*y - sigma*sigma)/(2*pi*pow(sigma,6)) *exp(-(x*x+y*y)/(2*sigma*sigma)); 
			matrixGxy[i][j] = (x*y)/(2*pi*pow(sigma,6)) *exp(-(x*x+y*y)/(2*sigma*sigma)); 
		}
	}

	Size s = image.size();
	// Assumption is that the text will not be on the edges of the picture
	int x_boundary = s.width  - filter_size; 
	int y_boundary = s.height - filter_size;

	// Looping over image cut by filter-size (avoiding fetching non existing pixels)
    for(int y=filter_size; y<y_boundary; y++) {	
    	for (int x=filter_size; x<x_boundary; x++) {   		
    		double sumGxx = 0; 
			double sumGyy = 0; 
			double sumGxy = 0; 		
    		// Calculating response of filters Gxx,Gyy,Gxy on the image
    		for(int l=-filter_size; l<=filter_size; l++){
    			for(int k=-filter_size; k<=filter_size; k++) {
    				sumGxx += (image.at<uchar>(y+l, x+k)*matrixGxx[k+filter_size][l+filter_size]); 
    				sumGyy += (image.at<uchar>(y+l, x+k)*matrixGyy[k+filter_size][l+filter_size]); 
    				sumGxy += (image.at<uchar>(y+l, x+k)*matrixGxy[k+filter_size][l+filter_size]);  
    			}
    		} 		

    		double thetaMax, thetaMin;
      		// Possible solutions for the equation of the maximum and minimum response
      		if (sumGyy-sumGxx == 0) {
        		thetaMax = pi/4;
      		} else if (sumGxy == 0) {
        		thetaMax = 0;
      		} else { 
        		double x = 2*sumGxy/(sumGyy-sumGxx);
        		thetaMax = 0.5*atan(x);
      		}
      
      		if (thetaMax > 0)
        		thetaMin = thetaMax-pi/2;
      		else
        		thetaMin = thetaMax+pi/2;
        	assert (thetaMax>=-pi/2 && thetaMax<=pi/2);

        	// After finding 2 angles which give maximal/minimal respone -> calculate response
      		double responseMax = calculateGTheta(sumGxx, sumGyy, sumGxy, thetaMax);
      		double responseMin = calculateGTheta(sumGxx, sumGyy, sumGxy, thetaMin);
      		// Swapping if needed
      		if (fabs(responseMax) < fabs(responseMin)){
        		swap(responseMax, responseMin);
        		swap(thetaMax, thetaMin);
      		}
      		filterImage.at<float>(y,x) = fabs(responseMax);
    	}
    }
}


vector<RotatedRect>  detectLetters(Mat &image) {
 	vector<RotatedRect> boundRectangles; 
 	Mat element;
 	
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3) );
    cv::morphologyEx(image, image, CV_MOP_CLOSE, element);
    vector< vector<Point> > contours; 
    cv::findContours(image, contours, 0, 1); 
    // Adding minAreaRect to vector or RotatedRects 
  	for( int i = 0; i < contours.size(); i++ ) { 
  		if(contours[i].size() > 100) {
  			RotatedRect r = minAreaRect(Mat(contours[i])); 
  			boundRectangles.push_back(r); 
  		}
  	}
    return boundRectangles;
 }


void processImage(Mat &image, string number) {

	// Image initialization 
	Mat resultImage(image.rows, image.cols, CV_32FC1); 
	Mat normImage(image.rows, image.cols, CV_8UC1); 
	Mat filterImage(image.rows, image.cols, CV_8UC1); 
	Mat threshold_image(image.rows, image.cols, CV_8UC1); 

	// sigma = 1.2 
	steerable_filter(image, resultImage, 1.2); 
	normalize(resultImage, normImage, 0, 255, NORM_MINMAX, -1);
	normImage.convertTo(filterImage, CV_8UC1);
	imshow("FilterImage", filterImage);
	bitwise_not(filterImage, filterImage); 

	int histogram[256]; 
	
	// Initialization of histogram
	for(int i=0; i<256; ++i) {
		histogram[i] = 0; 
	}

	// Calculating histogram of image
	for(int y=0; y < filterImage.rows; ++y) {
		for(int x=0; x < filterImage.cols; ++x) {
			histogram[ (int)filterImage.at<uchar>(y,x) ]++ ; 
		}
	}

	int total, histogram_threshold; 
	total = image.rows*image.cols; 
	histogram_threshold = findThreshold(histogram, total, 0.015);

	// Thresholding image by percentage of histogram 
	for(int y=0; y < filterImage.rows; ++y) { 
		for(int x=0; x < filterImage.cols; ++x) { 
			if( filterImage.at<uchar>(y,x) < histogram_threshold) { 
				threshold_image.at<uchar>(y,x) = 255; 
			} else {
				threshold_image.at<uchar>(y,x) = 0; 
			} 
		}
	}
	imshow("Threshold image", threshold_image); 
	dilate(threshold_image, threshold_image, Mat(), Point(-1, -1), 2, 1, 1);
	imshow("AfterDilate", threshold_image); 
	erode(threshold_image, threshold_image, Mat(), Point(-1, -1), 2, 1, 1);
	imshow("AfterErode", threshold_image);
	// Detecting letters
    vector<RotatedRect> boundRectangles = detectLetters(threshold_image);

  	Mat image_rgb(image.size(), CV_8UC3);
  	// convert grayscale to color image
  	cvtColor(image, image_rgb, CV_GRAY2RGB);
	
	// Displaying bounding rectangles
	for(int i=0; i<boundRectangles.size(); i++ ) {
		double width, height; 
		width = boundRectangles[i].size.width; 
		height = boundRectangles[i].size.height; 
		// Removing probably non-text rectangles
		if (width > 50 && height > 10) {
			Point2f rect_points[4]; 
			boundRectangles[i].points(rect_points); 
			for(int j=0; j<4; j++) {
				line(image_rgb, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255), 2, 8); 
			}
		}
	}    

    resultImage.release(); 
	normImage.release(); 
	filterImage.release(); 
	threshold_image.release();

    imshow("IzlazMetode", image_rgb);
    imwrite("Result_image" + number + ".png", image_rgb); 
}

/** Use of program : 
	Give path of image as parameter1 and number of resultimage as parametar 2
	Program will save result_image + parameter2.png in same folder
**/
int main(int argc, char **argv) {

	image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); 
	if(image.empty() ) {
		std::cerr << "Could not load given image!" << std::endl; 
	}
 
	processImage(image, argv[2]); 

	waitKey(0); 
	return 0; 
}