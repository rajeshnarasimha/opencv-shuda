//----------------------------
// Ground truth coordinates
//----------------------------

struct point3D
{
	point3D(float a, float b, float c) : x(a), y(b), z(c) {}
	float x, y, z;
};

struct point2D
{
	point2D(float a, float b) : x(a), y(b) {}
	float x, y;
};

struct rect2D
{
	rect2D(float a, float b, float c, float d) : x1(a), y1(b), x2(c), y2(d) {}
	float x1, y1, x2, y2;
};

// 3D "world" coordinates

const float paper_width  = 11.0f; // inch
const float paper_height =  8.5f; // inch

const float marker_margin = 1.0f; // margin between markers and paper, inch

const float dist_markers_x = paper_width  + 2*marker_margin; // inch 13
const float dist_markers_y = paper_height + 2*marker_margin; // inch 10.5

const float outer_margin = 0.5f; // margin around the markers, inch

const Eigen::Vector3f __eivCornersWorldCoordinates[4] = { // in 10 cm //of the whole image
	Eigen::Vector3f(-1.651f,-1.3335f, 0.f), Eigen::Vector3f(-1.651f, 1.3335, 0.f),
	Eigen::Vector3f( 1.651f, 1.3335f, 0.f), Eigen::Vector3f( 1.651f,-1.3335, 0.f)
};


// warped 2D image coordinates

const float DPI = 25.4f; // i.e. 1 pixel = 1 mm
const int dst_w = int((dist_markers_x+2*outer_margin)*DPI); // size of warped image ( fig 3 bottom row )
const int dst_h = int((dist_markers_y+2*outer_margin)*DPI);

const float x1__ =  outer_margin*DPI; 
const float x2__ = (outer_margin + dist_markers_x)*DPI;
const float y1__ =  outer_margin*DPI;
const float y2__ = (outer_margin + dist_markers_y)*DPI;

const point2D dst_corners[4] = { // coordinates of where the corners should be warped to, origin is located at up-left the x-axis pointing rightward and y pointing downward
	point2D(x1__,y1__), //point0 is up-left
	point2D(x1__,y2__), //point1 is bottom-left
	point2D(x2__,y2__), //point2 is bottom-right
	point2D(x2__,y1__)  //point2 is up-right
};

// area of paper
const float paper_margin = 0.1f; // inner margin (inch)
const rect2D planarROI(
	(outer_margin+marker_margin+paper_margin)*DPI,
	(outer_margin+marker_margin+paper_margin)*DPI,
	(outer_margin+marker_margin+paper_width -paper_margin)*DPI,
	(outer_margin+marker_margin+paper_height-paper_margin)*DPI
	);

/*
const point2D planarROI_p[4] = { // same coordinates as array of points
	point2D(planarROI.x1,planarROI.y1), point2D(planarROI.x2,planarROI.y1),
	point2D(planarROI.x2,planarROI.y2), point2D(planarROI.x1,planarROI.y2)
};*/

const Eigen::Vector3f __eivPlanarROIHomo[4] = { // in 10 cm //of the whole image
	Eigen::Vector3f( planarROI.x1, planarROI.y1, 1.0f), Eigen::Vector3f( planarROI.x2, planarROI.y1, 1.0f),
	Eigen::Vector3f( planarROI.x2, planarROI.y2, 1.0f), Eigen::Vector3f( planarROI.x1, planarROI.y2, 1.0f)
};

// area of texture allowed to use
const float texture_margin = 1.5f; // inner margin (inch)
const rect2D textureROI(
	(outer_margin+marker_margin+texture_margin)*DPI,
	(outer_margin+marker_margin+texture_margin)*DPI,
	(outer_margin+marker_margin+paper_width -texture_margin)*DPI,
	(outer_margin+marker_margin+paper_height-texture_margin)*DPI
	);
/*
const point2D textureROI_p[4] = { // same coordinates as array of points
	point2D(textureROI.x1,textureROI.y1), point2D(textureROI.x2,textureROI.y1),
	point2D(textureROI.x2,textureROI.y2), point2D(textureROI.x1,textureROI.y2)
};*/

const Eigen::Vector3f __eivTextureROIWorldHomo[4] = { // in 10 cm //of the whole image
	Eigen::Vector3f( textureROI.x1, textureROI.y1, 1.0f), Eigen::Vector3f( textureROI.x2, textureROI.y1, 1.0f),
	Eigen::Vector3f( textureROI.x2, textureROI.y2, 1.0f), Eigen::Vector3f( textureROI.x1, textureROI.y2, 1.0f)
};

const Eigen::Vector3f __eivTextureROI3D[4] = { // in 10 cm //of the whole image
	Eigen::Vector3f( textureROI.x1, textureROI.y1, .0f), Eigen::Vector3f( textureROI.x2, textureROI.y1, .0f),
	Eigen::Vector3f( textureROI.x2, textureROI.y2, .0f), Eigen::Vector3f( textureROI.x1, textureROI.y2, .0f)
};

