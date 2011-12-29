#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;

int main(int, char** argv)
{
	cv::FileStorage fs("test.xml", FileStorage::WRITE); // change "test.yml" you get yml format

	fs << "frameCount" << 5;
	time_t rawtime; 
	time(&rawtime);
	std::stringstream strDate;
	strDate << asctime(localtime(&rawtime));
	fs << "calibrationDate" << strDate.str(); //output to a string "Fri Jun 17 14:09:29 2011\n"
	cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	cv::Mat distCoeffs = (cv::Mat_<double>(5,1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
	fs << "features" << "[";
	for( int i = 0; i < 3; i++ )
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for( int j = 0; j < 8; j++ )
			fs << ((lbp >> j) & 1);
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();

	/////////////////////////////////////////////////////////////////

	FileStorage fs2("test.xml", FileStorage::READ);

	// first method: use (type) operator on FileNode.
	int frameCount = (int)fs2["frameCount"];

	std::string date;
	// second method: use FileNode::operator >>
	fs2["calibrationDate"] >> date;

	Mat cameraMatrix2, distCoeffs2;
	fs2["cameraMatrix"] >> cameraMatrix2;
	fs2["distCoeffs"] >> distCoeffs2;

	std::cout << "frameCount: " << frameCount << std::endl
		<< "calibration date: " << date << std::endl
		<< "camera matrix: " << cameraMatrix2 << std::endl
		<< "distortion coeffs: " << distCoeffs2 << std::endl;

	FileNode features = fs2["features"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	std::vector<uchar> lbpval;

	// iterate through a sequence using FileNodeIterator
	for( ; it != it_end; ++it, idx++ )
	{
		std::cout << "feature #" << idx << ": ";
		std::cout << "x=" << (int)(*it)["x"] << ", y=" << (int)(*it)["y"] << ", lbp: (";
		// you can also easily read numerical arrays using FileNode >> std::vector operator.
		(*it)["lbp"] >> lbpval;
		for( int i = 0; i < (int)lbpval.size(); i++ )
			std::cout << " " << (int)lbpval[i];
		std::cout << ")" << std::endl;
	}
	fs2.release();


	return 0;
}