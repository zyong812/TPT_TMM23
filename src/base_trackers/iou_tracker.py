from .utils import iou_xywh
from .tracker import Tracker


class IOUTracker(Tracker):
    """
    Intersection over Union Tracker.

    References
    ----------
    * Implementation of this algorithm is heavily based on https://github.com/bochinski/iou-tracker

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker.
        min_detection_confidence (float): Threshold for minimum detection confidence.
        max_detection_confidence (float): Threshold for max. detection confidence.
        iou_threshold (float): Intersection over union minimum value.
    """

    def __init__(
            self,
            max_lost=2,
            iou_threshold=0.5,
            tracker_output_format='visdrone_challenge'
    ):
        self.iou_threshold = iou_threshold

        super(IOUTracker, self).__init__(max_lost=max_lost, tracker_output_format=tracker_output_format)

    def update(self, bboxes, detection_scores, class_ids):
        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)
        self.frame_count += 1
        track_ids = list(self.tracks.keys())

        updated_tracks = []
        for track_id in track_ids:
            if len(detections) > 0:
                idx, best_match = max(enumerate(detections), key=lambda x: self.iou_and_class_match(self.tracks[track_id], x[1]))
                (bb, cid, scr) = best_match

                if self.iou_and_class_match(self.tracks[track_id], best_match) > self.iou_threshold:
                    self._update_track(track_id, self.frame_count, bb, scr, class_id=cid,
                                       iou_score=self.iou_and_class_match(self.tracks[track_id], best_match))
                    updated_tracks.append(track_id)
                    del detections[idx]

            if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        for bb, cid, scr in detections:
            self._add_track(self.frame_count, bb, scr, class_id=cid)

        outputs = self._get_tracks(self.tracks)
        return outputs

    def iou_and_class_match(self, track, det):
        if track.class_id == det[1]:
            return iou_xywh(track.bbox, det[0])
        else:
            return 0
