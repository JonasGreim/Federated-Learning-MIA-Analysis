def map_run_name(run_name):
    # Extract numeric part from e.g. "run42"
    try:
        run_id = int(run_name.replace("run", ""))
    except ValueError:
        return f"unknown ({run_name})"

    # IID groups
    if 42 <= run_id <= 45 or 50 <= run_id <= 53:
        return "IID"

    # non-IID groups
    elif 46 <= run_id <= 49 or 54 <= run_id <= 57:
        return "non_IID"

    return f"unknown ({run_name})"


