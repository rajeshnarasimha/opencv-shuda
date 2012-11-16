#ifndef PCL_CUDA_MARCHINGCUBES_HEADER
#define PCL_CUDA_MARCHINGCUBES_HEADER

namespace pcl { namespace device
{
	 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Marching cubes implementation

    /** \brief Binds marching cubes tables to texture references */
    void 
    bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);            
    
    /** \brief Unbinds */
    void 
    unbindTextures();
    
    /** \brief Scans tsdf volume and retrieves occuped voxes
      * \param[in] volume tsdf volume
      * \param[out] occupied_voxels buffer for occuped voxels. The function fulfills first row with voxel ids and second row with number of vertextes.
      * \return number of voxels in the buffer
      */
    int
    getOccupiedVoxels(/*const PtrStep<short2>&*/const cv::gpu::GpuMat& volume, DeviceArray2D<int>& occupied_voxels);

    /** \brief Computes total number of vertexes for all voxels and offsets of vertexes in final triangle array
      * \param[out] occupied_voxels buffer with occuped voxels. The function fulfills 3nd only with offsets      
      * \return total number of vertexes
      */
    int
    computeOffsetsAndTotalVertexes(DeviceArray2D<int>& occupied_voxels);

    /** \brief Generates final triangle array
      * \param[in] volume tsdf volume
      * \param[in] occupied_voxels occuped voxel ids (first row), number of vertexes(second row), offsets(third row).
      * \param[in] volume_size volume size in meters
      * \param[out] output triangle array            
      */
	void 
	generateTriangles (const cv::gpu::DevMem2D_<short2>& volume, const DeviceArray2D<int>& occupied_voxels, const float3& volume_size, DeviceArray<PointType>& output);

}//device
}//pcl

#endif