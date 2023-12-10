import re
import requests

def time_to_seconds(time_str):
    # Split the time string into its components
    parts = time_str.split(':')
    parts = [int(p) for p in parts]

    # Calculate total seconds based on the number of parts (hours, minutes, seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    else:
        return 0  # Return 0 if the format is incorrect

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def parse_chapters(description):
    # Split the description into lines
    lines = description.split("\\n")

    # Define the regular expression for matching timestamps
    regex = re.compile(r'(\d{0,2}:?\d{1,2}:\d{2})')

    chapters = []

    for line in lines:
        # Find all matches of the regex in the line
        matches = regex.findall(line)

        if matches:
            timestamp = matches[0]
            # Removing the timestamp from the line to get the title
            title = ' '.join([word for word in line.split() if timestamp not in word])

            chapters.append({
                "timestamp": time_to_seconds(timestamp),
                "title": title
            })

    return chapters