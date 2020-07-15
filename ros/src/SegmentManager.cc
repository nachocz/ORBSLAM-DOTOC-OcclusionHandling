#include "SegmentManager.h"
namespace DEF_OBJ_TRACK
{
SegmentManager::SegmentManager()
{
}
SegmentManager::~SegmentManager()
{
}

void SegmentManager::SegmentManagerUpdate(std::map<uint32_t, std::shared_ptr<Segment>>  &segment_list_now)
{
    if (segment_manager_history_size_ < max_segment_manager_history_size_)
    {
        segment_manager_history_size_++;
    }

    std::map<uint32_t, std::map<uint32_t, std::shared_ptr<Segment>>> new_segment_history_;

    new_segment_history_[1] = segment_list_now;

    for (int i = 2; i <= segment_manager_history_size_; i++)
    {
        new_segment_history_[i] = segment_history_[i - 1];
    }

    segment_history_.clear();
    segment_history_ = new_segment_history_;
}

pcl::PointXYZ SegmentManager::RGBtoLAB(pcl::PointXYZRGBA &colorRGB)
{
    pcl::PointXYZ colorLAB, colorXYZ;

    double r = colorRGB.x / 255.0;
    double g = colorRGB.y / 255.0;
    double b = colorRGB.z / 255.0;

    r = ((r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : (r / 12.92)) * 100.0;
    g = ((g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : (g / 12.92)) * 100.0;
    b = ((b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : (b / 12.92)) * 100.0;

    colorXYZ.x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    colorXYZ.y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    colorXYZ.z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    double colorXYZx = colorXYZ.x / 95.047;
    double colorXYZy = colorXYZ.y / 100.00;
    double colorXYZz = colorXYZ.z / 108.883;

    colorXYZx = (colorXYZx > 0.008856) ? cbrt(colorXYZx) : (7.787 * colorXYZx + 16.0 / 116.0);
    colorXYZy = (colorXYZy > 0.008856) ? cbrt(colorXYZy) : (7.787 * colorXYZy + 16.0 / 116.0);
    colorXYZz = (colorXYZz > 0.008856) ? cbrt(colorXYZz) : (7.787 * colorXYZz + 16.0 / 116.0);

    colorLAB.x = (116.0 * colorXYZy) - 16;      // l
    colorLAB.y = 500 * (colorXYZx - colorXYZy); // a
    colorLAB.z = 200 * (colorXYZy - colorXYZz); // b

    return colorLAB;
}

bool SegmentManager::doesItBelongToSegment(pcl::PointXYZRGBA &SVColor, const pcl::PointNormal &SVNormal, std::shared_ptr<Segment> comparisonSegment, pcl::visualization::PCLVisualizer::Ptr viewer)
{

    bool it_belongs_to_segment = false;

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    pcl::PointNormal sv_centroid;
    pcl::PointXYZ sv_centroidRadiusSearch;

    sv_centroidRadiusSearch.x = SVNormal.x;
    sv_centroidRadiusSearch.y = SVNormal.y;
    sv_centroidRadiusSearch.z = SVNormal.z;

    Eigen::Vector4f normal_sv;

    normal_sv[0] = SVNormal.normal_x;
    normal_sv[1] = SVNormal.normal_y;
    normal_sv[2] = SVNormal.normal_z;

    pcl::PointXYZ SVColorLAB = RGBtoLAB(SVColor);

    float distanceRejection = 0;
    float colorRejection = 0;
    float normalRejection = 0;
    float nodeDegreeRejection = 0;

    int interactorTotalDegree = 0;
    int interactionTotalNeighbours = 0;

    float centroidDistanceTotalWeighted = 0.0;
    float colorCIELABdistanceTotalWeighted;
    float normalDistanceTotalWeighted;

    //checking if it has any object's sv within its exploration radius
    if ((comparisonSegment->Segments_sv_AdjacencyOctree_)->radiusSearch(sv_centroidRadiusSearch, comparisonSegment->exploration_radius_, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {

        float centroidNumerator = 0;
        long colorNumerator = 0;
        float normalNumerator = 0;
        int totalDegree = 0;

        for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
        {

            int node_degree = comparisonSegment->nodes_degrees_[pointIdxRadiusSearch[i] + 1];

            pcl::PointNormal neighbour_normal = comparisonSegment->segments_normals_[pointIdxRadiusSearch[i] + 1];
            pcl::PointXYZRGBA neighbour_color = comparisonSegment->segments_colors_RGB_[pointIdxRadiusSearch[i] + 1];
            pcl::PointXYZ neighbour_colorLAB = RGBtoLAB(neighbour_color);

            float centroidDistance = pcl::geometry::distance(SVNormal, neighbour_normal);
            float colorCIELABdistance = pcl::geometry::distance(SVColorLAB, neighbour_colorLAB);

            Eigen::Vector4f normal_neighbour;

            normal_neighbour[0] = neighbour_normal.normal_x;
            normal_neighbour[1] = neighbour_normal.normal_y;
            normal_neighbour[2] = neighbour_normal.normal_z;

            float AngularDistance = (1.0f / PI) * std::acos(normal_sv.dot(normal_neighbour));

            centroidNumerator = centroidNumerator + (comparisonSegment->exploration_radius_ - centroidDistance) * node_degree;

            colorNumerator = colorNumerator + colorCIELABdistance * node_degree;
            normalNumerator = normalNumerator + AngularDistance * node_degree;
            totalDegree = totalDegree + node_degree;
        }
        float averageTotalDegree = totalDegree / pointIdxRadiusSearch.size();

        centroidDistanceTotalWeighted = (centroidNumerator / (comparisonSegment->max_graph_degree_ * comparisonSegment->exploration_radius_)); // * (centroidNumerator / (comparisonSegment->max_graph_degree_ * comparisonSegment->exploration_radius_)); //The larger the better
        colorCIELABdistanceTotalWeighted = colorNumerator / (totalDegree * centroidDistanceTotalWeighted);                //The larger the worse
        normalDistanceTotalWeighted = normalNumerator / (totalDegree * centroidDistanceTotalWeighted);                    //The larger the worse

        if ((colorCIELABdistanceTotalWeighted <= colorThreshold) && (normalDistanceTotalWeighted <= normalThreshold))
        {
            it_belongs_to_segment = true;
        }
    }

    return it_belongs_to_segment;
}
} // namespace DEF_OBJ_TRACK