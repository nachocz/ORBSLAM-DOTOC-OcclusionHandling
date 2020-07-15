#include "IncludesAndDefinitions.h"
#include "Segment.h"

namespace DEF_OBJ_TRACK
{
class Segment;
class SegmentManager
{
public:
    SegmentManager();
    ~SegmentManager();
    SegmentManager(std::shared_ptr<Segment> &segment_list_prev_);

    void SegmentManagerUpdate(std::map<uint32_t, std::shared_ptr<Segment>> &segment_list_now);
    bool doesItBelongToSegment(pcl::PointXYZRGBA &SVColor, const pcl::PointNormal &SVNormal, std::shared_ptr<Segment> comparisonSegment, pcl::visualization::PCLVisualizer::Ptr viewer);
    pcl::PointXYZ RGBtoLAB(pcl::PointXYZRGBA &colorRGB);

public:
    //Basic elements

    //Number of past frames saved
    float segment_manager_history_size_;
    float max_segment_manager_history_size_;

    //Number of segments in frame k
    float number_of_segments_in_frame_;

    std::map<uint32_t, std::shared_ptr<Segment>> segment_list_prev_;
    std::map<uint32_t, std::shared_ptr<Segment>> segment_list_now_;

    //Segments
    std::map<uint32_t, std::shared_ptr<Segment>> segment_list_;
    std::map<uint32_t, std::map<uint32_t, std::shared_ptr<Segment>>> segment_history_;

    float colorThreshold, normalThreshold;

private:
};

} // namespace DEF_OBJ_TRACK
