/****************************************************************************

 http://www.libqglviewer.com - contact@libqglviewer.com
 
*****************************************************************************/



class Viewer : public QGLViewer
{
public:
	Viewer();
    ~Viewer();
protected :
    virtual void draw();
    virtual void init();
    virtual QString helpString() const;
	//virtual void resizeEvent( QResizeEvent * event);
	virtual void keyPressEvent(QKeyEvent *e);
	virtual void mousePressEvent(QMouseEvent *e	);
	virtual void mouseReleaseEvent(QMouseEvent *e );
	virtual void mouseMoveEvent( QMouseEvent *e );
	virtual void wheelEvent(QWheelEvent *e );

	void loadFromYml();
	void reset();
	void exportToYml(const std::string& strFileName_);
	void importGroundTruth(const std::string& strFileName_);
	void getNextFrame();
	void other();
	void loadTexture ( const cv::Mat& cvmImg_, GLuint* puTexture_ );
	void renderTexture( const GLuint uTexture_ );
	void creatGroundTruthMask(cv::Mat* pcvmMask_);
	cv::Mat _cvmHomographyAll;
	std::vector<Eigen::Matrix3f> _veimHomography;
	Eigen::Matrix3f _eimHomoCurr;
	Eigen::Vector3f _eivPixelH[4]; 

	cv::Mat _cvmMaskCurr;
	cv::Mat _cvmMaskPrev;

	btl::video::VideoSource::tp_shared_ptr _pVideo;
	btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
	btl::kinect::CKeyFrame::tp_scoped_ptr _pPrevFrame;

	//btl::image::semidense::CSemiDenseTrackerOrb::tp_scoped_ptr _pTracker;
	btl::image::CTrackerSimpleFreak::tp_scoped_ptr _pTracker;

	cv::gpu::GpuMat _cvgmRGB;
	boost::scoped_ptr<cv::gpu::GpuMat> _pcvgmColorGraph;
	GLuint _uTexture;

	ushort _uResolution;
	ushort _uPyrHeight;
	Eigen::Vector3f _eivCw;
	bool _bUseNIRegistration;
	ushort _uCubicGridResolution;
	float _fVolumeSize;
	int _nMode;//btl::kinect::VideoSourceKinect::PLAYING_BACK
	std::string _oniFileName; // the openni file 
	std::string _strVideoFileName;
	bool _bRepeat;// repeatedly play the sequence 
	int _nRecordingTimeInSecond;
	float _fTimeLeft;
	int _nStatus;//1 restart; 2 //recording continue 3://pause 4://dump
	bool _bDisplayImage;
	bool _bLightOn;
	bool _bRenderReference;
	bool _bTrack;

	unsigned int _uFrameIdx;
	int nStatus;

	SoccerPitch _pitchProcessor;
	cv::Mat _cvmPlayField;
};