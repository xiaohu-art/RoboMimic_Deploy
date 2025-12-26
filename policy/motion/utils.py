import re
import numpy as np
from typing import Any, Sequence

def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray):
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is 4.
        vec: The vector in (x, y, z). Shape is 3.

    Returns:
        The rotated vector in (x, y, z). Shape is 3.
    """
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2
    return (vec - quat[0:1] * t + np.cross(xyz, t))

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)

def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return quat_conjugate(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec + quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray = None, q02: np.ndarray = None
):
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

def resolve_matching_names(
    keys, list_of_strings, preserve_order: bool = False
):
    """Match a list of query regular expressions against a list of strings and return the matched indices and names.

    When a list of query regular expressions is provided, the function checks each target string against each
    query regular expression and returns the indices of the matched strings and the matched strings.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query regular expressions.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query regular expressions.

    For example, consider the list of strings is ['a', 'b', 'c', 'd', 'e'] and the regular expressions are ['a|c', 'b'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings and the
    strings as: ([0, 1, 2], ['a', 'b', 'c']). When :attr:`preserve_order` is True, it will return them as:
    ([0, 2, 1], ['a', 'c', 'b']).

    Note:
        The function does not sort the indices. It returns the indices in the order they are found.

    Args:
        keys: A regular expression or a list of regular expressions to match the strings in the list.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.

    Returns:
        A tuple of lists containing the matched indices and names.

    Raises:
        ValueError: When multiple matches are found for a string in the list.
        ValueError: When not all regular expressions are matched.
    """
    # resolve name keys
    if isinstance(keys, str):
        keys = [keys]
    # find matching patterns
    index_list = []
    names_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one regular expression
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(keys))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if re.fullmatch(re_key, potential_match_string):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                # add for regex key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
    # check that all regular expressions are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(keys, keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
        )
    # return
    return index_list, names_list

joint_names_mujoco = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

joint_names_isaac = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint"
]

body_names_mujoco = [
    'pelvis', 
    'left_hip_pitch_link', 
    'left_hip_roll_link', 
    'left_hip_yaw_link', 
    'left_knee_link', 
    'left_ankle_pitch_link', 
    'left_ankle_roll_link', 
    'right_hip_pitch_link', 
    'right_hip_roll_link', 
    'right_hip_yaw_link', 
    'right_knee_link', 
    'right_ankle_pitch_link', 
    'right_ankle_roll_link', 
    'waist_yaw_link', 
    'waist_roll_link', 
    'torso_link', 
    'left_shoulder_pitch_link', 
    'left_shoulder_roll_link', 
    'left_shoulder_yaw_link', 
    'left_elbow_link', 
    'left_wrist_roll_link', 
    'left_wrist_pitch_link', 
    'left_wrist_yaw_link', 
    'right_shoulder_pitch_link', 
    'right_shoulder_roll_link', 
    'right_shoulder_yaw_link', 
    'right_elbow_link', 
    'right_wrist_roll_link', 
    'right_wrist_pitch_link', 
    'right_wrist_yaw_link'
]

body_names_isaac = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link"
]