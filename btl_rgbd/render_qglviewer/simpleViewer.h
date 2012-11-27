/****************************************************************************

 http://www.libqglviewer.com - contact@libqglviewer.com
 
*****************************************************************************/

#include <QGLViewer/qglviewer.h>

class Viewer : public QGLViewer
{
public:
	Viewer();
    ~Viewer();
protected :
    virtual void draw();
    virtual void init();
    virtual QString helpString() const;
	virtual void resizeEvent( QResizeEvent * event);
	virtual void keyPressEvent(QKeyEvent *e);
	virtual void mousePressEvent(QMouseEvent *e	);
	virtual void mouseReleaseEvent(QMouseEvent *e );
	virtual void mouseMoveEvent( QMouseEvent *e );


	void loadFromYml();
	void drawLogo() const;
	btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
	btl::gl_util::CGLUtil::tp_shared_ptr _pGL;



	ushort _uResolution;
	ushort _uPyrHeight;
	Eigen::Vector3f _eivCw;
	bool _bUseNIRegistration;
	ushort _uCubicGridResolution;
	float _fVolumeSize;
	int _nMode;//btl::kinect::VideoSourceKinect::PLAYING_BACK
	std::string _oniFileName; // the openni file 
	bool _bRepeat;// repeatedly play the sequence 
	int _nRecordingTimeInSecond;
	float _fTimeLeft;
	int _nStatus;//1 restart; 2 //recording continue 3://pause 4://dump
	bool _bDisplayImage;
	bool _bLightOn;
	bool _bRenderReference;

};
