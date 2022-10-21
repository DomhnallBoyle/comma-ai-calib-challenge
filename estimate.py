import argparse

import cv2
import numpy as np


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


def get_corner_points(frame, num_corner_points):
    corner_points = cv2.goodFeaturesToTrack(
        frame_to_grey(frame=frame), 
        maxCorners=num_corner_points, 
        qualityLevel=0.1, 
        minDistance=10,
        blockSize=7
    )
    # assert len(corner_points) == num_corner_points, f'{len(corner_points)} != {num_corner_points}'

    return corner_points


def calculate_optical_flow(current_frame, next_frame, current_points):
    # Lucas-Kanade method
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        frame_to_grey(current_frame), 
        frame_to_grey(next_frame), 
        prevPts=current_points, 
        nextPts=None, 
        # winSize=(15, 15),
        # maxLevel=2,
        # criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    assert len(current_points) == len(next_points) == len(status)

    return next_points, status, error


def main(args):
    # From paper: A Robust Road Vanishing Point Detection Adapted to the Real-World Driving Scenes
    video_path = 'labelled/0.hevc'
    gt_path = 'labelled/0.txt'
    num_frames = 200

    video_frames = get_video_frames(video_path=video_path, num_frames=num_frames)
    euler_angles = get_euler_angles(gt_path=gt_path)[:num_frames]
    assert len(video_frames) == len(euler_angles)

    num_corner_points, k, min_vector_distance, max_motion_vectors = 500, 1, 2, 400
    motion_vectors = []

    i = 0
    current_frame = video_frames[i]  # initial frame
    height, width = current_frame.shape[:2]
    centre_point = (int(height / 2), int(width / 2))
    current_points = None
    skip_corner_points = False
    point_diff = lambda p1, p2: (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    while True:
        if i + k == len(video_frames): 
            break

        if not skip_corner_points:
            # detect corner points of frame
            _current_points = get_corner_points(frame=current_frame, num_corner_points=num_corner_points)
            if current_points is None:
                current_points = _current_points
            else:
                current_points = np.concatenate((current_points, _current_points), axis=0)

        # track every corner point into the next frame
        next_points, status, error = calculate_optical_flow(
            current_frame=current_frame, 
            next_frame=video_frames[i + k], 
            current_points=current_points
        )
        
        # tracked = 1, untracked = 0
        tracked_indices = np.where(status.flatten() == 1)[0]  # returns indices
        untracked_indices = np.where(status.flatten() == 0)[0]

        # calculate euclidean distance of corresponding tracked points
        new_tracked_indices = []
        for j in tracked_indices:
            point_1 = current_points[j]
            point_2 = next_points[j]  # head point
            distance = np.linalg.norm(point_1 - point_2)
            if distance > min_vector_distance:
                motion_vectors.append([point_1, point_2])
                new_tracked_indices.append(j)
        tracked_points = next_points[new_tracked_indices]

        # not enough motion vectors, update the current frame and points
        if len(motion_vectors) <= max_motion_vectors:
            current_frame = video_frames[i + k]
            current_points = tracked_points
            i += 1
            k = 1
            skip_corner_points = False
            continue

        # filter out type 2 motion vectors = accelerating
        motion_vectors = [v for v in motion_vectors if point_diff(v[1][0], centre_point) > point_diff(v[0][0], centre_point)]

        # filter out type 3 motion vectors = decelerating vehicles

        if args.debug:
            for vector in motion_vectors: 
                x1, y1 = vector[0].astype(np.int32)[0]
                x2, y2 = vector[1].astype(np.int32)[0]
                cv2.line(current_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.imshow('Frame', current_frame)
            cv2.waitKey(25)

        k += 1
        skip_corner_points = True
        motion_vectors = []

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
