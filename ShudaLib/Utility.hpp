#ifndef BTL_UTILITY_HEADER
#define BTL_UTILITY_HEADER


#include "OtherUtil.hpp"
#include "EigenUtil.hpp"
#include "CVUtil.hpp"
#include "Converters.hpp"

#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

namespace btl
{
namespace utility
{

template< class T >
void normalEstimationGL( const T* pDepth_, const cv::Mat& cvmRGB_, std::vector<const unsigned char*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_ )
{
	BTL_ASSERT(cvmRGB_.type()== CV_8UC3, "CVUtil::normalEstimationGL() Error: the input must be a 3-channel color image.")
	vColor_->clear();
	vPt_->clear();
	vNormal_->clear();

	unsigned char* pColor_ = (unsigned char*) cvmRGB_.data;

	Eigen::Vector3d n1, n2, n3, v(0,0,1);

	//calculate normal
	for( unsigned int r = 0; r < cvmRGB_.rows; r++ )
		for( unsigned int c = 0; c < cvmRGB_.cols; c++ )
		{
			// skip the right and bottom boarder line
			if( c == cvmRGB_.cols-1 || r == cvmRGB_.rows-1 )
			{
				pColor_+=3;
				continue;
			}
			size_t i;
			i = r*cvmRGB_.cols + c;
			size_t ii = i*3;
			Eigen::Vector3d pti  ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
			size_t i1;
			i1 = i + 1;
			ii = i1*3;
			Eigen::Vector3d pti1 ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
			size_t j1;
			j1 = i + cvmRGB_.cols;
			ii = j1*3;
			Eigen::Vector3d ptj1 ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );

			if( fabs( pti(2) ) > 0.0000001 && fabs( pti1(2) ) > 0.0000001 && fabs( ptj1(2) ) > 0.0000001 )
			{
				n1 = pti1 - pti;
				n2 = ptj1 - pti;
				n3 = n1.cross(n2);
				n3.normalize();
				if ( v.dot(n3) < 0 )
				{
					//PRINT( n3 );
					n3 = -n3;
				}
				vColor_->push_back(pColor_);
				vPt_->push_back(pti);
				vNormal_->push_back(n3);
			}
			pColor_+=3;
		}
		return;
}

template< class T >
void normalEstimationGLPCL( const T* pDepth_, const cv::Mat& cvmRGB_, int nKNearest_, std::vector<const unsigned char*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_ )
{
	
	vColor_->clear();
	vPt_->clear();
	vNormal_->clear();

	const unsigned char* pColor_ = (const unsigned char*)cvmRGB_.data;

	pcl::PointCloud<pcl::PointXYZ> _cloudNoneZero;
	pcl::PointCloud<pcl::Normal>   _cloudNormals;
	for( unsigned int r = 0; r < cvmRGB_.rows; r++ )
	for( unsigned int c = 0; c < cvmRGB_.cols; c++ )
	{
		size_t i;
		i = r*cvmRGB_.cols + c;
		size_t ii = i*3;
		if( pDepth_[ii+2] > 0.0000001 )
		{
			pcl::PointXYZ point( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
			_cloudNoneZero.push_back(point);
			vColor_->push_back( pColor_ );
		}
		pColor_+=3;
	}

	//get normal using pcl
	pcl::search::KdTree<pcl::PointXYZ>::Ptr pTree (new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

	// Estimate point normals
	ne.setSearchMethod (pTree);
	ne.setInputCloud (_cloudNoneZero.makeShared());
	ne.setKSearch (nKNearest_);
	ne.compute (_cloudNormals);

	// collect the points and normals

	for (size_t i = 0; i < _cloudNoneZero.points.size (); ++i)
	{
		Eigen::Vector3d eivPt  ( _cloudNoneZero.points[i].x,_cloudNoneZero.points[i].y,_cloudNoneZero.points[i].z );
		Eigen::Vector3d eivNl  ( _cloudNormals.points[i].normal_x,_cloudNormals.points[i].normal_y,_cloudNormals.points[i].normal_z );
		vPt_->push_back(eivPt);
		vNormal_->push_back(eivNl);
	}
	return;
}

}//utility
}//btl
#endif