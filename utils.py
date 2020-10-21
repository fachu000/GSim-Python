


def time_to_str(time_delta):

    hours = time_delta.seconds // 3600
    reminder = time_delta.seconds % 3600
    minutes = reminder // 60
    seconds = (time_delta.seconds - hours * 3600 -
               minutes * 60) + time_delta.microseconds / 1e6
    time_str = ""
    if time_delta.days:
        time_str = "%d days, " % time_delta.days
    if hours:
        time_str = time_str + "%d hours, " % hours
    if minutes:
        time_str = time_str + "%d minutes, " % minutes
    if time_str:
        time_str = time_str + "and "

    return time_str + "%.3f seconds" % seconds
    #set_trace()
