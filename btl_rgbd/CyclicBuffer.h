#ifndef BTL_NI_BUFFER
#define BTL_NI_BUFFER


namespace btl{ namespace kinect{
#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
	printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
	return rc;													\
}
#define CHECK_RC_ERR(rc, what, errors)			\
	{												\
	if (rc == XN_STATUS_NO_NODE_PRESENT)		\
	{											\
	XnChar strError[1024];					\
	errors.ToString(strError, 1024);		\
	printf("%s\n", strError);				\
}											\
	CHECK_RC(rc, what)							\
}


// The cyclic buffer, to which frames will be added and from where they will be dumped to files
class CCyclicBuffer
{
public:
	typedef boost::scoped_ptr< CCyclicBuffer > tp_scoped_ptr;
	// Creation - set the OpenNI objects
	CCyclicBuffer(xn::Context& context, xn::DepthGenerator& depthGenerator, xn::ImageGenerator& imageGenerator) :
	  m_context(context), m_depthGenerator(depthGenerator), m_imageGenerator(imageGenerator), m_pFrames(NULL)
	  {
		  m_nNextWrite = 0;
		  m_nBufferSize = 0;
		  m_nBufferCount = 0;
	  }
	  // Initialization - set outdir and time of each recording
	  void Initialize(XnChar* strDirName, XnUInt32 nSeconds)
	  {
		  xnOSStrCopy(m_strDirName, strDirName, XN_FILE_MAX_PATH);
		  m_nBufferSize = nSeconds*30;
		  m_pFrames = XN_NEW_ARR(SingleFrame, m_nBufferSize);
	  }
	  // Save new data from OpenNI
	void Update(const xn::DepthMetaData& DepthMD_, const xn::ImageMetaData& ImageMD_)
	{
		// Save latest depth frame
		m_pFrames[m_nNextWrite].depthFrame.CopyFrom(DepthMD_);
		// Save latest image frame
		m_pFrames[m_nNextWrite].imageFrame.CopyFrom(ImageMD_);

		// See if buffer is already full
		if (m_nBufferCount < m_nBufferSize)
		{
			m_nBufferCount++;
		}
		// Make sure cyclic buffer pointers are good
		m_nNextWrite++;
		if (m_nNextWrite == m_nBufferSize)
		{
			m_nNextWrite = 0;
		}
	}

	  // Save the current state of the buffer to a file
	  XnStatus Dump()
	  {
		  xn::MockDepthGenerator mockDepth;
		  xn::MockImageGenerator mockImage;

		  xn::EnumerationErrors errors;
		  XnStatus rc;

		  // Create recorder
		  rc = m_context.CreateAnyProductionTree(XN_NODE_TYPE_RECORDER, NULL, m_recorder, &errors);
		  CHECK_RC_ERR(rc, "Create recorder", errors);

		  // Create name of new file
		  time_t rawtime;
		  struct tm *timeinfo;
		  time(&rawtime);
		  timeinfo = localtime(&rawtime);
		  XnChar strFileName[XN_FILE_MAX_PATH];
		  sprintf(strFileName, "%s/%04d%02d%02d-%02d%02d%02d.oni", m_strDirName,
			  timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

		  m_recorder.SetDestination(XN_RECORD_MEDIUM_FILE, strFileName);
		  printf("Creating file %s\n", strFileName);

			// Create mock nodes based on the depth generator, to save depth
			rc = m_context.CreateMockNodeBasedOn(m_depthGenerator, NULL, mockDepth);		  CHECK_RC(rc, "Create depth node");
			rc = m_recorder.AddNodeToRecording(mockDepth, XN_CODEC_16Z_EMB_TABLES);			  CHECK_RC(rc, "Add depth node");
			// Create mock nodes based on the image generator, to save image
			rc = m_context.CreateMockNodeBasedOn(m_imageGenerator, NULL, mockImage);		  CHECK_RC(rc, "Create image node");
			rc = m_recorder.AddNodeToRecording(mockImage, XN_CODEC_JPEG);					  CHECK_RC(rc, "Add image node");

		  // Write frames from next index (which will be next to be written, and so the first available)
		  // this is only if a full loop was done, and this frame has meaningful data
		  if (m_nNextWrite < m_nBufferCount)
		  {
			  // Not first loop, right till end
			  for (XnUInt32 i = m_nNextWrite; i < m_nBufferSize; ++i)
			  {
				  mockDepth.SetData(m_pFrames[i].depthFrame);
				  mockImage.SetData(m_pFrames[i].imageFrame);
				  m_recorder.Record();
			  }
		  }
		  // Write frames from the beginning of the buffer to the last on written
		  for (XnUInt32 i = 0; i < m_nNextWrite; ++i)
		  {
			  mockDepth.SetData(m_pFrames[i].depthFrame);
			  mockImage.SetData(m_pFrames[i].imageFrame);
			  m_recorder.Record();
		  }

		  // Close recorder
		  m_recorder.Release();

		  return XN_STATUS_OK;
	  }//Dump

protected:
	struct SingleFrame
	{
		xn::DepthMetaData depthFrame;
		xn::ImageMetaData imageFrame;
	};

	SingleFrame* m_pFrames;
	XnUInt32 m_nNextWrite;
	XnUInt32 m_nBufferSize;
	XnUInt32 m_nBufferCount;
	XnChar m_strDirName[XN_FILE_MAX_PATH];

	xn::Context& m_context;
	xn::DepthGenerator& m_depthGenerator;
	xn::ImageGenerator& m_imageGenerator;
	xn::Recorder m_recorder;

private:
	XN_DISABLE_COPY_AND_ASSIGN(CCyclicBuffer);
};

}//kinect
}//btl

#endif