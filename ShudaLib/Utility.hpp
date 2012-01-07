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
template< class T1, class T2 >
void normalEstimationGLPCL( const T1* pDepth_, const T2* pColor_, const unsigned int& uRows_, const unsigned int& uCols_, std::vector<const T2*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_ )
{
	vColor_->clear();
	vPt_->clear();
	vNormal_->clear();

	pcl::PointCloud<pcl::PointXYZ> _cloudNoneZero;
	pcl::PointCloud<pcl::Normal>   _cloudNormals;
	for( unsigned int r = 0; r < uRows_; r++ )
		for( unsigned int c = 0; c < uCols_; c++ )
		{
			size_t i;
			i = r*uCols_ + c;
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
		ne.setKSearch (6);
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