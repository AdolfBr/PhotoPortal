def select_camera_id(available_ids, saved_id):
    """Return the saved camera ID when available, otherwise the first ID."""
    if not available_ids:
        return None
    if saved_id in available_ids:
        return saved_id
    return available_ids[0]
