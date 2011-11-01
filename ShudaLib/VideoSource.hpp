#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCE
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCE

#include <btl/Image/Image.hpp>
#include <stdexcept>

namespace btl
{
namespace extra
{
namespace videosource
{

class VideoSource
{

   public:

      VideoSource(){}

      virtual ~VideoSource(){}

      Eigen::Vector2i frameSize() const;

      virtual const ImageRegionConstRGB getNextFrame() = 0;

      const ImageRegionConstRGB getCurrentFrame();

      struct Exception : public std::runtime_error
      {
         Exception(const std::string& str) : std::runtime_error(str) {}
      };

   protected:

      ImageRGB _frame;

      Eigen::Vector2i _frameSize;

};

} //namespace videosource
} //namespace extra
} //namespace btl


namespace btl
{
namespace extra
{

using videosource::VideoSource;

} //namespace extra
} //namespace btl


// ====================================================================
// === Implementation

namespace btl
{
namespace extra
{
namespace videosource
{

//inline VideoSource::VideoSource() {}

//inline VideoSource::~VideoSource() {}

inline Eigen::Vector2i VideoSource::frameSize() const
{
   return _frameSize;
}

inline const ImageRegionConstRGB VideoSource::getCurrentFrame()
{
   if(_frame.null()) return getNextFrame();
   return ImageRegionConstRGB(_frame);
}

} //namespace videosource
} //namespace extra
} //namespace btl


#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCE
