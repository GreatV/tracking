#include "byte_tracker.h"
#include "lapjv.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer) {
    track_thresh = 0.5;
    high_thresh = 0.6;
    match_thresh = 0.8;

    frame_id = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
    std::cout << "Init ByteTrack!" << std::endl;
}

BYTETracker::~BYTETracker()
= default;

std::vector<STrack> BYTETracker::update(std::vector<Object> &objects) {

    ////////////////// Step 1: Get detections //////////////////
    this->frame_id++;
    std::vector<STrack> activated_stracks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> removed_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> detections;
    std::vector<STrack> detections_low;

    std::vector<STrack> detections_cp;
    std::vector<STrack> tracked_stracks_swap;
    std::vector<STrack> resa, resb;
    std::vector<STrack> output_stracks;

    std::vector<STrack *> unconfirmed;
    std::vector<STrack *> tracked_stracks;
    std::vector<STrack *> strack_pool;
    std::vector<STrack *> r_tracked_stracks;

    if (!objects.empty()) {
        for (auto &object: objects) {
            std::vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = static_cast<float>(object.x);
            tlbr_[1] = static_cast<float>(object.y);
            tlbr_[2] = static_cast<float>(object.x + object.w);
            tlbr_[3] = static_cast<float>(object.y + object.h);

            float score = object.prob;

            STrack state_track(STrack::tlbr_to_tlwh(tlbr_), score);
            if (score >= track_thresh)
                detections.push_back(state_track);
            else
                detections_low.push_back(state_track);
        }
    }

    // Add newly detected tracklets to tracked_stracks
    for (auto &tracked_strack: this->tracked_stracks) {
        if (!tracked_strack.is_activated)
            unconfirmed.push_back(&tracked_strack);
        else
            tracked_stracks.push_back(&tracked_strack);
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
    STrack::multi_predict(strack_pool, this->kalman_filter);

    std::vector<std::vector<float> > dists;
    int dist_size = 0, dist_size_size = 0;
    dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

    std::vector<std::vector<int> > matches;
    std::vector<int> u_track, u_detection;
    linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

    for (auto &match: matches) {
        STrack *track = strack_pool[match[0]];
        STrack *det = &detections[match[1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    for (int i: u_detection) {
        detections_cp.push_back(detections[i]);
    }
    detections.clear();
    detections.assign(detections_low.begin(), detections_low.end());

    for (int i: u_track) {
        if (strack_pool[i]->state == TrackState::Tracked) {
            r_tracked_stracks.push_back(strack_pool[i]);
        }
    }

    dists.clear();
    dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (auto &match: matches) {
        STrack *track = r_tracked_stracks[match[0]];
        STrack *det = &detections[match[1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    for (int i: u_track) {
        STrack *track = r_tracked_stracks[i];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    std::vector<int> u_unconfirmed;
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (auto &match: matches) {
        unconfirmed[match[0]]->update(detections[match[1]], this->frame_id);
        activated_stracks.push_back(*unconfirmed[match[0]]);
    }

    for (int i: u_unconfirmed) {
        STrack *track = unconfirmed[i];
        track->mark_removed();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i: u_detection) {
        STrack *track = &detections[i];
        if (track->score < this->high_thresh)
            continue;
        track->activate(this->kalman_filter, this->frame_id);
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (auto &lost_strack: this->lost_stracks) {
        if (this->frame_id - lost_strack.end_frame() > this->max_time_lost) {
            lost_strack.mark_removed();
            removed_stracks.push_back(lost_strack);
        }
    }

    for (auto &tracked_strack: this->tracked_stracks) {
        if (tracked_strack.state == TrackState::Tracked) {
            tracked_stracks_swap.push_back(tracked_strack);
        }
    }
    this->tracked_stracks.clear();
    this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
    this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

    //std::cout << activated_stracks.size() << std::endl;

    this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
    for (auto &lost_strack: lost_stracks) {
        this->lost_stracks.push_back(lost_strack);
    }

    this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
    for (auto &removed_strack: removed_stracks) {
        this->removed_stracks.push_back(removed_strack);
    }

    remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

    this->tracked_stracks.clear();
    this->tracked_stracks.assign(resa.begin(), resa.end());
    this->lost_stracks.clear();
    this->lost_stracks.assign(resb.begin(), resb.end());

    for (auto &tracked_strack: this->tracked_stracks) {
        if (tracked_strack.is_activated) {
            output_stracks.push_back(tracked_strack);
        }
    }
    return output_stracks;
}

std::vector<STrack *>
BYTETracker::joint_stracks(std::vector<STrack *> &track_list_a, std::vector<STrack> &track_list_b) {
    std::map<int, int> exists;
    std::vector<STrack *> res;
    for (auto &i: track_list_a) {
        exists.insert(std::pair<int, int>(i->track_id, 1));
        res.push_back(i);
    }
    for (auto &i: track_list_b) {
        int tid = i.track_id;
        if (!exists[tid] || exists.count(tid) == 0) {
            exists[tid] = 1;
            res.push_back(&i);
        }
    }
    return res;
}

std::vector<STrack> BYTETracker::joint_stracks(std::vector<STrack> &track_list_a, std::vector<STrack> &track_list_b) {
    std::map<int, int> exists;
    std::vector<STrack> res;
    for (auto &i: track_list_a) {
        exists.insert(std::pair<int, int>(i.track_id, 1));
        res.push_back(i);
    }
    for (auto &i: track_list_b) {
        int tid = i.track_id;
        if (!exists[tid] || exists.count(tid) == 0) {
            exists[tid] = 1;
            res.push_back(i);
        }
    }
    return res;
}

std::vector<STrack> BYTETracker::sub_stracks(std::vector<STrack> &track_list_a, std::vector<STrack> &track_list_b) {
    std::map<int, STrack> stracks;
    for (auto &i: track_list_a) {
        stracks.insert(std::pair<int, STrack>(i.track_id, i));
    }
    for (auto &i: track_list_b) {
        int tid = i.track_id;
        if (stracks.count(tid) != 0) {
            stracks.erase(tid);
        }
    }

    std::vector<STrack> res;
    std::map<int, STrack>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it) {
        res.push_back(it->second);
    }

    return res;
}

void BYTETracker::remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb,
                                           std::vector<STrack> &stracksa, std::vector<STrack> &stracksb) {
    std::vector<std::vector<float> > pdist = iou_distance(stracksa, stracksb);
    std::vector<std::pair<int, int> > pairs;
    for (size_t i = 0; i < pdist.size(); i++) {
        for (size_t j = 0; j < pdist[i].size(); j++) {
            if (pdist[i][j] < 0.15) {
                pairs.emplace_back(i, j);
            }
        }
    }

    std::vector<int> dupa, dupb;
    for (auto &pair: pairs) {
        int timep = stracksa[pair.first].frame_id - stracksa[pair.first].start_frame;
        int timeq = stracksb[pair.second].frame_id - stracksb[pair.second].start_frame;
        if (timep > timeq)
            dupb.push_back(pair.second);
        else
            dupa.push_back(pair.first);
    }

    for (size_t i = 0; i < stracksa.size(); i++) {
        auto iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }

    for (size_t i = 0; i < stracksb.size(); i++) {
        auto iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }
}

void BYTETracker::linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size,
                                    int cost_matrix_size_size, float thresh,
                                    std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a,
                                    std::vector<int> &unmatched_b) {
    if (cost_matrix.empty()) {
        for (int i = 0; i < cost_matrix_size; i++) {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++) {
            unmatched_b.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol;
    std::vector<int> colsol;
    // double c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
    lapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (int i = 0; i < static_cast<int>(rowsol.size()); i++) {
        if (rowsol[i] >= 0) {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < static_cast<int>(colsol.size()); i++) {
        if (colsol[i] < 0) {
            unmatched_b.push_back(i);
        }
    }
}

std::vector<std::vector<float> >
BYTETracker::ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs) {
    std::vector<std::vector<float> > ious;
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    ious.resize(atlbrs.size());
    for (auto &iou: ious) {
        iou.resize(btlbrs.size());
    }

    //bbox_ious
    for (size_t k = 0; k < btlbrs.size(); k++) {
        std::vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
        for (size_t n = 0; n < atlbrs.size(); n++) {
            float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0) {
                float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if (ih > 0) {
                    float ua =
                            (atlbrs[n][2] - atlbrs[n][0] + 1) * (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                } else {
                    ious[n][k] = 0.0;
                }
            } else {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

std::vector<std::vector<float> >
BYTETracker::iou_distance(std::vector<STrack *> &atracks, std::vector<STrack> &btracks, int &dist_size,
                          int &dist_size_size) {
    std::vector<std::vector<float> > cost_matrix;
    if (atracks.empty() || btracks.empty()) {
        dist_size = static_cast<int>(atracks.size());
        dist_size_size = static_cast<int>(btracks.size());
        return cost_matrix;
    }
    std::vector<std::vector<float> > atlbrs, btlbrs;
    for (auto &atrack: atracks) {
        atlbrs.push_back(atrack->tlbr);
    }
    for (auto &btrack: btracks) {
        btlbrs.push_back(btrack.tlbr);
    }

    dist_size = static_cast<int>(atracks.size());
    dist_size_size = static_cast<int>(btracks.size());

    std::vector<std::vector<float> > _ious = ious(atlbrs, btlbrs);

    for (auto &i: _ious) {
        std::vector<float> _iou;
        for (float j: i) {
            _iou.push_back(1 - j);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

std::vector<std::vector<float> > BYTETracker::iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks) {
    std::vector<std::vector<float> > atlbrs, btlbrs;
    for (auto &atrack: atracks) {
        atlbrs.push_back(atrack.tlbr);
    }
    for (auto &btrack: btracks) {
        btlbrs.push_back(btrack.tlbr);
    }

    std::vector<std::vector<float> > _ious = ious(atlbrs, btlbrs);
    std::vector<std::vector<float> > cost_matrix;
    for (auto &i: _ious) {
        std::vector<float> _iou;
        for (float j: i) {
            _iou.push_back(1 - j);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

double
BYTETracker::lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
                   bool extend_cost, float cost_limit, bool return_cost) {
    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    int n_rows = static_cast<int>(cost.size());
    int n_cols = static_cast<int>(cost[0].size());
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols) {
        n = n_rows;
    } else {
        if (!extend_cost) {
            std::cout << "set extend_cost=True" << std::endl;
            auto ret = system("pause");
            exit(ret);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX) {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (auto &i: cost_c_extended)
            i.resize(n);

        if (cost_limit < LONG_MAX) {
            for (auto &i: cost_c_extended) {
                for (float &j: i) {
                    j = cost_limit / 2.0;
                }
            }
        } else {
            float cost_max = -1;
            for (auto &i: cost_c) {
                for (float j: i) {
                    if (j > cost_max)
                        cost_max = j;
                }
            }
            for (auto &i: cost_c_extended) {
                for (float &j: i) {
                    j = cost_max + 1;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++) {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++) {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int *x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0) {
        std::cout << "Calculate Wrong!" << std::endl;
        ret = system("pause");
        exit(ret);
    }

    double opt = 0.0;

    if (n != n_rows) {
        for (int i = 0; i < n; i++) {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++) {
            colsol[i] = y_c[i];
        }

        if (return_cost) {
            for (size_t i = 0; i < rowsol.size(); i++) {
                if (rowsol[i] != -1) {
                    //cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    } else if (return_cost) {
        for (size_t i = 0; i < rowsol.size(); i++) {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++) {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}

cv::Scalar BYTETracker::get_color(int idx) {
    idx += 3;
    return {static_cast<double>(37 * idx % 255), static_cast<double>(17 * idx % 255),
            static_cast<double>(29 * idx % 255)};
}
