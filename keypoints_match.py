import cv2

import front_end


def match_frames(frame_1, frame_2, method, ratio_threshold=None):
    if ratio_threshold is None:
        ratio_threshold = front_end.track_ratio(method)
    if method == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2) # if the method is SIFT, use the L2 norm to match the descriptors.
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) # if the method is ORB, use the Hamming norm to match the descriptors.

    descriptors_1 = frame_1.Descriptors # get the descriptors of the first frame.
    descriptors_2 = frame_2.Descriptors # get the descriptors of the second frame.
    if descriptors_1 is None or descriptors_2 is None:
        return [] # if the descriptors are None, return an empty list.
    ''' use knn by default, kept this for testing purposes only'''
    if frame_2.match_type == "BF":
        return bf.match(descriptors_1, descriptors_2) # if the match type is BF, use the brute force matcher to match the descriptors.

    knn = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = []
    for pair in knn:
        # OpenCV can return only one neighbor for some descriptors.
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance: #if m is a much better match than n, add m to the good matches.
            good_matches.append(m)

    return good_matches


# Backward compatibility with previous function name.
Match = match_frames
