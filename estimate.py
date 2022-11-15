import argparse
import random
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from frechetdist import frdist

FOCAL_LENGTH = 910


def get_video_frames(video_path, num_frames=None): 
    video_capture = cv2.VideoCapture(video_path)
    frames = []

    while True:
        if num_frames and len(frames) == num_frames: 
            break

        success, frame = video_capture.read()
        if not success: 
            break
        frames.append(frame)

    video_capture.release()

    return np.asarray(frames)


def get_euler_angles(gt_path):
    return np.loadtxt(gt_path)


def frame_to_grey(frame): 
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_corner_points(frame, num_corner_points, corner_quality_level):
    corner_points = cv2.goodFeaturesToTrack(
        frame_to_grey(frame=frame), 
        maxCorners=num_corner_points,  # returns strongest if more than found
        qualityLevel=corner_quality_level,  # minimum accepted quality (% of max quality corner)
        minDistance=10  # min euclidean distance between corners
    )

    return corner_points


def calculate_optical_flow(current_frame, next_frame, current_points):
    # Lucas-Kanade method
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        frame_to_grey(current_frame), 
        frame_to_grey(next_frame), 
        prevPts=current_points, 
        nextPts=None
    )
    assert len(current_points) == len(next_points) == len(status)

    return next_points, status, error


def euclidean_distance(p1, p2): 
    return np.linalg.norm(p1 - p2)


def vector_norm(v):
    return np.linalg.norm(v)


def point_diff(p1, p2):
    return p1 - p2


def horizon_angle(v): 
    x, y = point_diff(*v)
    
    return np.arctan(y / x) * 180 / np.pi


def calculate_extended_point(v, l):
    head_point, _ = v
    u = point_diff(*v)
    v_norm = vector_norm(v)

    head_point_extended = (head_point + (l * (u / v_norm))).astype(int)

    return head_point_extended


def pitch_yaw_to_vp(pitch, yaw, k):
    vp_cam_x = np.tan(yaw)
    vp_cam_y = -np.tan(pitch / np.cos(yaw))

    return np.round(k.dot(np.array([vp_cam_x, vp_cam_y, 1]))[:2]).astype(int)


def vp_to_pitch_yaw(vp, k_inv): 
    input_shape = vp.shape
    vp = np.atleast_2d(vp)
    vp = np.hstack((vp, np.ones((vp.shape[0], 1))))
    cam_pts = vp.dot(k_inv.T)
    cam_pts[(vp < 0).any(axis=1)] = np.nan
    cam_pts = cam_pts[:, :2].reshape(input_shape)

    yaw_calib = np.arctan(cam_pts[0])
    pitch_calib = np.arctan(-cam_pts[1]) * np.cos(yaw_calib)

    return pitch_calib, yaw_calib


def angle_between_vectors(v1, v2):
    v1, v2 = point_diff(*v1), point_diff(*v2)
    v1_norm, v2_norm = vector_norm(v1), vector_norm(v2)

    angle_rad = np.arccos(np.dot(v1 / v1_norm, v2 / v2_norm))

    return angle_rad * 180 / np.pi


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return np.array((x, y)).astype(int)


def ransac_vp(motion_vectors, max_inlier_angle, max_best_score_attempts):
    best_vp, best_vp_score = None, -np.inf
    attempts_since_improvement = 0
    while attempts_since_improvement < max_best_score_attempts:
        
        if len(motion_vectors) < 2: 
            return

        # get random VP as initial VP i.e. hypothesis
        num_vp_attempts = 10
        while True:
            v1, v2 = random.sample(motion_vectors, 2)
            try:
                vp = line_intersection(v1, v2)
            except Exception:
                num_vp_attempts -= 1
                if num_vp_attempts == 0: 
                    return
                continue

            break

        vp_score = 0
        for v in motion_vectors:
            # create vector from base point to VP
            head, base = v
            u = [head, vp]

            theta = angle_between_vectors(v, u)

            # exponential score - vector with smaller angle contributes more to the voting
            score = np.exp(-abs(theta)) if theta < max_inlier_angle else 0
            vp_score += score

        if vp_score > best_vp_score:
            best_vp_score = vp_score
            best_vp = vp
            attempts_since_improvement = 0
        else:
            attempts_since_improvement += 1  # no improvement

    return best_vp


class VideoBuffer:

    def __init__(self, video_path, early_stop=None):
        self.video_capture = cv2.VideoCapture(video_path)
        self.early_stop = early_stop
        self.frames = []

    def __getitem__(self, i):
        while True:
            if self.frames:
                try:
                    return True, self.frames[i]
                except IndexError:
                    return False, None

            time.sleep(1)

    def __len__(self):
        return len(self.frames)

    def load_frames(self):
        i = 0
        while True:
            success, frame = self.video_capture.read()
            if not success:
                break
            self.frames.append(frame)
            
            i += 1
            if self.early_stop == i: 
                break

        self.video_capture.release()

    def start(self):
        t = Thread(target=self.load_frames, args=())
        t.start()


class MVClustering:
    
    def __init__(self, num_clusters, frame_width, frame_height, max_iterations=100):
        self.num_clusters = num_clusters
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_iterations = max_iterations
        self.random_point = lambda: np.array([random.randint(0, self.frame_width), random.randint(0, self.frame_height)])
        self.initialise_centroids()

    def initialise_centroids(self):
        # centroids are lines
        self.centroids = [[self.random_point(), self.random_point()] for _ in range(self.num_clusters)]

    def run(self, motion_vectors):
        # TODO: stop when centroids don't change
        while self.max_iterations > 0: 
            clusters = {i: [] for i in range(self.num_clusters)}

            # calculate distance of all mvs to all centroids
            # assign mv to closest centroid
            for v in motion_vectors:
                best_distance, closest_centroid = np.inf, None
                for j, centroid in enumerate(self.centroids): 
                    distance = frdist(v, centroid)
                    if distance < best_distance: 
                        best_distance = distance
                        closest_centroid = j
                clusters[closest_centroid].append(v)

            # calculate new centroids - average of assigned motion vectors
            new_centroids = []
            for i, vectors in clusters.items():
                average_p1, average_p2 = [0, 0], [0, 0]
                for p1, p2 in vectors: 
                    average_p1 = np.add(average_p1, p1)
                    average_p2 = np.add(average_p2, p2)
                new_centroid = [
                    np.divide(average_p1, len(vectors)), 
                    np.divide(average_p2, len(vectors))
                ]
                new_centroids.append(new_centroid)
            self.centroids = new_centroids
            
            self.max_iterations -= 1
        
        final_centroids = []
        for v in self.centroids: 
            if np.isnan(v).any():
                continue
            v = [v[0].astype(int), v[1].astype(int)]
            final_centroids.append(v)

        return final_centroids


def main(args): 
    # TODO: ignore bottom percentage of frame

    video_path = Path(f'{args.video_directory}/{args.video_id}.hevc')
    gt_path = Path(f'{args.video_directory}/{args.video_id}.txt')
    assert video_path.exists()

    video_buffer = VideoBuffer(video_path=str(video_path), early_stop=args.early_stop)
    video_buffer.start()

    euler_angles = get_euler_angles(gt_path=gt_path) if gt_path.exists() else None

    t, k = 0, 1
    p_t0 = None
    motion_vectors = {}
    detect_corner_points = True

    video_frame = video_buffer[0][1]
    height, width = video_frame.shape[:2]
    centre_point = np.array([width / 2, height / 2]).astype(int)
    camera_intrinsics = np.array([
        [FOCAL_LENGTH, 0.0, width / 2],
        [0.0, FOCAL_LENGTH, height / 2],
        [0.0, 0.0, 1.0]
    ])
    camera_intrinsics_inv  = np.linalg.inv(camera_intrinsics)
    line_boundaries = [
        [(0, 0), (width, 0)],
        [(0, height), (width, height)],
        [(0, 0), (0, height)],
        [(width, 0), (width, height)]
    ]  # line boundaries of the frame

    while True:
        success, video_frame = video_buffer[t]
        if not success: 
            break

        # get corner points of initial frame
        if detect_corner_points:
            corner_points = get_corner_points(
                frame=video_frame, 
                num_corner_points=args.num_corner_points, 
                corner_quality_level=args.corner_quality_level
            )
            # update the set of corner points
            if p_t0 is None:
                p_t0 = corner_points
            else:
                p_t0 = np.concatenate((p_t0, corner_points), axis=0)

        success, next_frame = video_buffer[t + k]
        if not success: 
            break

        # track every point onto the next frame
        next_points, status, error = calculate_optical_flow(
            current_frame=video_frame, 
            next_frame=next_frame, 
            current_points=p_t0
        )

        tracked_indices = np.where(status.flatten() == 1)[0]  # returns indices
        indices_to_remove = np.where(status.flatten() == 0)[0].tolist()  # untracked points

        # update all tracked points onto set p_t0k_tracked
        p_t0k_tracked = [next_points[i] for i in tracked_indices]

        # calculate euclidean distance between 
        if not motion_vectors.get(t + k):
            motion_vectors[t + k] = []
        for i, p in zip(tracked_indices, p_t0k_tracked):
            p1 = p.flatten().astype(int)  # head point (tracked point)
            p2 = p_t0[i].flatten().astype(int)  # base point
            distance = euclidean_distance(p1=p1, p2=p2)
            if distance > args.displacement_distance:
                motion_vectors[t + k].append([p1, p2])  # head, base
            else:
                indices_to_remove.append(i)

        # delete untracked points from p_t0
        p_t0 = np.delete(p_t0, indices_to_remove, axis=0)

        if len(motion_vectors[t + k]) <= args.max_motion_vectors:
            t += k
            k = 1
            detect_corner_points = True
            continue

        num_vectors_before_filtering = len(motion_vectors[t + k])

        # exclude type 2 motion vectors
        motion_vectors[t + k] = [
            v for v in motion_vectors[t + k] 
            if euclidean_distance(calculate_extended_point(v=v, l=args.extended_l_pixels), centre_point) > euclidean_distance(v[1], centre_point)
        ]
        num_vectors_after_type_2_filtering = len(motion_vectors[t + k])

        # exclude type 3 motion vectors
        motion_vectors[t + k] = [
            v for v in motion_vectors[t + k]
            if not (-args.min_horizon_angle <= horizon_angle(v=v) <= args.min_horizon_angle)
            and euclidean_distance(*v) > args.min_vector_distance
        ]
        num_vectors_after_type_3_filtering = len(motion_vectors[t + k])

        if args.debug: 
            print(f'Filtering: Total ({num_vectors_before_filtering}), Type 2 ({num_vectors_after_type_2_filtering}), Type 3 ({num_vectors_after_type_3_filtering})')
            
        k += 1
        detect_corner_points = False

    # # cluster the motion vectors of every frame
    # clustering = MVClustering(
    #     num_clusters=10, 
    #     frame_width=width,
    #     frame_height=height
    # )
    # new_motion_vectors = {}
    # for i, _motion_vectors in motion_vectors.items():
    #     _motion_vectors = clustering.run(motion_vectors=_motion_vectors)
    #     new_motion_vectors[i] = _motion_vectors
    #     clustering.initialise_centroids()
    # motion_vectors = new_motion_vectors

    pitches, yaws = [], []
    num_rolling_frames = 10
    for i, motion_vectors in motion_vectors.items():
        video_frame = video_buffer[i][1]

        # extending the motion vectors to the boundaries of the frame i.e. get intersection points with boundaries
        new_motion_vectors = []
        for j, v in enumerate(motion_vectors):
            new_v = []            
            for line in line_boundaries:
                try:
                    x_int, y_int = line_intersection(v, line)
                except Exception:
                    continue

                if 0 <= x_int <= width and 0 <= y_int <= height:
                    new_v.append(np.array([x_int, y_int]).astype(int))

            if len(new_v) == 2:
                new_motion_vectors.append(new_v)

                # draw first and last motion vectors
                if args.debug and j in [0, len(motion_vectors) - 1]:
                    cv2.line(video_frame, tuple(new_v[0]), tuple(new_v[1]), (255, 255, 255), thickness=2)
                    cv2.line(video_frame, tuple(v[0]), tuple(v[1]), (0, 0, 255), thickness=2)
        motion_vectors = new_motion_vectors

        # estimate the VP using a RANSAC method
        estimated_vp = ransac_vp(
            motion_vectors=motion_vectors, 
            max_inlier_angle=args.max_inlier_angle, 
            max_best_score_attempts=args.max_best_score_attempts
        )

        if estimated_vp is not None:
            pitch, yaw = vp_to_pitch_yaw(estimated_vp, k_inv=camera_intrinsics_inv)
            if not np.isnan(pitch) and not np.isnan(yaw):
                pitches.append(pitch)
                yaws.append(yaw)

        # rolling average pitch and yaw
        start = 0
        if len(pitches) > num_rolling_frames:
            start = len(pitches) - num_rolling_frames
        pitch, yaw = np.mean(pitches[start:]), np.mean(yaws[start:])

        estimated_vp = pitch_yaw_to_vp(pitch, yaw, k=camera_intrinsics)

        # draw gt VP, estimated VP and frame centre
        if args.debug:
            points = [centre_point, estimated_vp]
            if euler_angles is not None:
                gt_vp = pitch_yaw_to_vp(*euler_angles[i], k=camera_intrinsics)
                points.append(gt_vp)

            for point, colour in zip(points, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                cv2.circle(video_frame, tuple(point), 5, colour, -1)

            cv2.imshow('Video', video_frame)
            cv2.waitKey(args.fps)
    
        print(pitch, yaw)

    print(pitch, yaw)  # last frame

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_directory')
    parser.add_argument('video_id', type=int)
    parser.add_argument('--early_stop', type=int)
    parser.add_argument('--num_corner_points', type=int, default=500)
    parser.add_argument('--corner_quality_level', type=float, default=0.01)
    parser.add_argument('--displacement_distance', type=int, default=2)
    parser.add_argument('--max_motion_vectors', type=int, default=400)
    parser.add_argument('--extended_l_pixels', type=int, default=10)
    parser.add_argument('--min_horizon_angle', type=int, default=10)
    parser.add_argument('--min_vector_distance', type=int, default=25)
    parser.add_argument('--max_inlier_angle', type=int, default=45)
    parser.add_argument('--max_best_score_attempts', type=int, default=50)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
