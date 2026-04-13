def check_overlap(win_start, win_end, seiz_start, seiz_end):
    """
    Checks if two intervals [win_start, win_end] and [seiz_start, seiz_end] overlap.
    """
    return max(win_start, seiz_start) < min(win_end, seiz_end)

def get_label_for_window(start_sec, end_sec, seizures):
    """
    Assigns binary label based on overlap with given seizure intervals.
    1 = seizure, 0 = non-seizure.
    """
    for seiz in seizures:
        seiz_start, seiz_end = seiz
        if check_overlap(start_sec, end_sec, seiz_start, seiz_end):
            return 1
    return 0

import random

def label_windows(windows, balance_ratio=3.0):
    """
    Given a list of window metadata tuples (record_info, start, end),
    assign a label to each and return updated list of dicts.
    Under-samples background (0) to balance_ratio * seizure count.
    """
    seizure_windows = []
    background_windows = []
    for record, start, end in windows:
        label = get_label_for_window(start, end, record['seizures'])
        item = {
            'record': record,
            'start': start,
            'end': end,
            'label': label
        }
        if label == 1:
            seizure_windows.append(item)
        else:
            background_windows.append(item)
            
    if balance_ratio > 0 and len(seizure_windows) > 0:
        target_bg_count = int(len(seizure_windows) * balance_ratio)
        if len(background_windows) > target_bg_count:
            random.seed(42)
            background_windows = random.sample(background_windows, target_bg_count)
            
    labeled_data = seizure_windows + background_windows
    random.shuffle(labeled_data)
    
    return labeled_data
