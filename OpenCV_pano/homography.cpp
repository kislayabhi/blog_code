#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/stitching.hpp"
#include <algorithm>

using namespace cv;
using namespace std;

Mat translateImg(Mat &img, int offsetx, int offsety)
{
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);

	warpAffine(img, img, trans_mat, Size(3 * img.cols, 3 * img.rows)); // 3,4 is usual
	return trans_mat;
}

void warp_crops(Mat& im_1, const Mat& im_2)
{
	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();


	// Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect( im_1, keypoints_1 );
	f2d->detect( im_2, keypoints_2 );

	// Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute( im_1, keypoints_1, descriptors_1 );
	f2d->compute( im_2, keypoints_2, descriptors_2 );

	// Step 3: Matching descriptor vectors using BFMatcher :
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	// Keep best matches only to have a nice drawing.
	// We sort distance between descriptor matches
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;

	for (int i = 0; i < 200; i++)
		bestMatches.push_back(matches[index.at < int > (i, 0)]);


	// 1st image is the destination image and the 2nd image is the src image
	std::vector<Point2f> dst_pts;                   //1st
	std::vector<Point2f> source_pts;                //2nd

	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
		cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
		//-- Get the keypoints from the good matches
		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
	}

	// Mat img_matches;
	// drawMatches( im_1, keypoints_1, im_2, keypoints_2,
	//           bestMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
	//           vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//
	// //-- Show detected matches
	// imwrite( "Good_Matches.jpg", img_matches );



	Mat H = findHomography( source_pts, dst_pts, CV_RANSAC );
	cout << H << endl;

	Mat wim_2;
	warpPerspective(im_2, wim_2, H, im_1.size());

	for (int i = 0; i < im_1.cols; i++)
		for (int j = 0; j < im_1.rows; j++) {
			Vec3b color_im1 = im_1.at<Vec3b>(Point(i, j));
			Vec3b color_im2 = wim_2.at<Vec3b>(Point(i, j));
			if (norm(color_im1) == 0)
				im_1.at<Vec3b>(Point(i, j)) = color_im2;

		}

}

int main( int argc, char** argv)
{
	int width = 640 * 2;
	int height = 480 * 2;

	// Read in the image.
	Mat im_1 = imread("../data/11.jpg");

	resize(im_1, im_1, Size(width, height));

	translateImg(im_1, 800, 1000); // 2000 is usual

	Mat im_2 = imread("../data/12.jpg");
	resize(im_2, im_2, Size(width, height));

	Mat im_3 = imread("../data/13.jpg");
	resize(im_3, im_3, Size(width, height));

	Mat im_4 = imread("../data/14.jpg");
	resize(im_4, im_4, Size(width, height));

	warp_crops(im_1, im_2);
	warp_crops(im_1, im_3);
	warp_crops(im_1, im_4);

	namedWindow("translated 1st image", 0);
	imshow("translated 1st image", im_1);
	waitKey(0);

	imwrite("result.jpg", im_1);
}
