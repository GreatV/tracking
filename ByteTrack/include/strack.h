#ifndef TRACKING_STRACK_H
#define TRACKING_STRACK_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "kalman_filter.h"


enum TrackState {
    New = 0, Tracked, Lost, Removed
};

class STrack {
public:
    STrack(std::vector<float> tlwh_, float score);

    ~STrack();

    std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);

    void static multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);

    void static_tlwh();

    void static_tlbr();

    static std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);

    std::vector<float> to_xyah() const;

    void mark_lost();

    void mark_removed();

    static int next_id();

    int end_frame() const;

    void activate(byte_kalman::KalmanFilter &filter, int id);

    void re_activate(STrack &new_track, int id, bool new_id = false);

    void update(STrack &new_track, int id);

public:
    bool is_activated;
    int track_id;
    int state;

    std::vector<float> _tlwh;
    std::vector<float> tlwh;
    std::vector<float> tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};

#endif //TRACKING_STRACK_H
