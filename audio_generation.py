import yt_dlp as youtube_dl
from pydub import AudioSegment
import os
from bs4 import BeautifulSoup
import json
import argparse
import subprocess
import tempfile
import csv
import math
import pandas as pd

# List of classes to iterate over
CLASSES = [
    "Baby cry, infant cry",
    "Music",
    "Speech",
    "Water",
    "Fire alarm",
    "Television",
    "Animal",
    "Tick-tock",
    "Drawer open or close",
    "Writing",
    "Vehicle",
    "Door",
    "Rail transport",
    "Dishes, pots, and pans",
    "Wood"
]

# List of class combinations based on Granger Causality
class_combinations = [
    ("Television", "Speech"),
    ("Television", "Rail transport"),
    ("Television", "Vehicle"),
    ("Rail transport", "Vehicle"),
    ("Drawer open or close", "Wood"),
    ("Drawer open or close", "Door"),
    ("Baby cry, infant cry", "Animal"),
    ("Water", "Vehicle"),
    ("Television", "Fire alarm"),
    ("Fire alarm", "Water"),
    ("Speech", "Music"),  
    ("Television", "Drawer open or close"),
    ("Rail transport", "Fire alarm"),
    ("Television", "Door"),
    ("Door", "Vehicle")
]

# Define equivalence labels for specific classes
equivalence_labels = {
    "rail_transport": ["rail_transport", "train"],#, "railroad_car", "train_wagon"], tuve un error al escribir la etiqueta "railroad_car, train_wagon" y lamentablemente ha quedado excluida
    "animal": ["animal", "dog", "domestic_animals_pets"],
    "speech": ["speech", "male_speech_man_speaking", "narration_monologue", "conversation", "female_speech_woman_speaking"],
    "tick_tock": ["tick"],
    "dishes_pots_pans": ["dishes_pots_pans", "cutlery_silverware"],
    "television": ["television", "music", "speech"],
    "fire_alarm": ["fire_alarm", "smoke_detector_smoke_alarm"],
    "drawer_open_close": ["drawer_open_close", "inside_small_room"]
}

def sanitize_class_name(class_name):
    """
    Sanitizes the class name to be filesystem-safe and lowercase.

    Args:
        class_name (str): Original class name.

    Returns:
        str: Sanitized class name.
    """
    sanitized = class_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('/', '_')
    return sanitized.lower()

def setup_directory(class_dir):
    """
    Creates the directory for a class if it does not exist.

    Args:
        class_dir (str): Path to the class directory.
    """
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"Directory '{class_dir}' created.")
    else:
        print(f"Directory '{class_dir}' already exists.")

def download_youtube_audio(yt_url, output_filename, class_dir, cookies_path=None):
    """
    Downloads the audio from a YouTube video using yt_dlp.

    Args:
        yt_url (str): YouTube video URL.
        output_filename (str): Base name for the output file.
        class_dir (str): Directory where the audio will be saved.
        cookies_path (str, optional): Path to the cookies file for authentication.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(class_dir, f'{output_filename}.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading audio from {yt_url}")
            ydl.download([yt_url])
            print(f"Download completed: {os.path.join(class_dir, output_filename + '.mp3')}")
            return True
        except youtube_dl.utils.DownloadError as e:
            print(f"Error downloading {yt_url}: {e}")
            return False

def extract_video_metadata(html_filepath, target_class):
    """
    Extracts video metadata from an HTML file and stores it in a list.

    Args:
        html_filepath (str): Path to the HTML file.
        target_class (str): The main class being searched for.

    Returns:
        list: List of dictionaries containing metadata.
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        print(f"HTML file '{html_filepath}' loaded successfully.")
    except Exception as e:
        print(f"Error loading HTML file: {e}")
        return []

    metadata_list = []
    SEGMENT_DURATION = 5  # Duration of the segment in seconds

    # Get equivalent labels for the main class
    equivalents = equivalence_labels.get(target_class, [])
    allowed_labels = set([target_class] + equivalents)
    print(f"Allowed labels for class '{target_class}': {allowed_labels}")

    # Extract videos
    for video in soup.find_all('div', {'class': 'u'}):
        try:
            # Get video ID
            yt_id = video.get('data-ytid')
            if not yt_id:
                print("Video without 'data-ytid' found. Skipping.")
                continue

            # Get start and end times
            start_time = float(video.get('data-start', 0))
            end_time = float(video.get('data-end', 0))

            # Verify valid times
            if start_time == 0 and end_time == 0:
                print(f"Warning: Invalid start/end times for video {yt_id}. Skipping.")
                continue

            # Get labels
            data_labels = video.get('data-labels')
            if not data_labels:
                print(f"Video {yt_id} without labels. Skipping.")
                continue

            # Decode JSON labels
            labels_json = json.loads(data_labels.replace('&quot;', '"'))

            # Extract label names
            labels = [label[0].strip().lower() for label in labels_json]

            # Create video URL
            video_url = f"https://www.youtube.com/watch?v={yt_id}"

            # Check allowed labels
            if target_class.lower() in labels and all(label in allowed_labels for label in labels):
                # Calculate processed times
                processed_start_time = start_time + 1  # Start 1 second after original
                processed_end_time = processed_start_time + SEGMENT_DURATION

                # Ensure end time does not exceed original
                if processed_end_time > end_time:
                    processed_end_time = end_time

                # Create metadata entry
                metadata_entry = {
                    'video_url': video_url,
                    'target_class': target_class,
                    'original_labels': ';'.join(labels),
                    'start_time': processed_start_time,
                    'end_time': processed_end_time
                }

                metadata_list.append(metadata_entry)
                print(f"Included video {yt_id}: {labels}")
            else:
                print(f"Excluded video {yt_id} due to disallowed labels: {labels}")

        except Exception as e:
            print(f"Error processing video entry: {e}")
            continue

    return metadata_list

def save_metadata_to_csv(metadata_list, output_file):
    """
    Saves the list of metadata to a CSV file.

    Args:
        metadata_list (list): List of dictionaries containing metadata.
        output_file (str): Output CSV file name.
    """
    if not metadata_list:
        print("No metadata to save.")
        return
        
    fieldnames = ['video_url', 'target_class', 'original_labels', 'start_time', 'end_time']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_list)
        print(f"Metadata saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving metadata to CSV: {e}")

def process_class_metadata(class_name, html_filepath, class_dir):
    """
    Processes metadata for a specific class and saves it to the class directory.

    Args:
        class_name (str): Name of the class.
        html_filepath (str): Path to the HTML file with video data.
        class_dir (str): Directory of the class where metadata will be saved.
    """
    print(f"\nProcessing metadata for class: {class_name}")
    
    # Ensure class directory exists
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"Directory '{class_dir}' created.")
    
    # Generate the filename using the class name
    sanitized_class = sanitize_class_name(class_name)
    output_file = os.path.join(class_dir, f"{sanitized_class}_metadata.csv")
    
    metadata_list = extract_video_metadata(html_filepath, class_name.lower())
    save_metadata_to_csv(metadata_list, output_file)

def extract_valid_seconds(segment, min_dbfs=-60):
    """
    Filters and extracts valid seconds from an audio segment based on a dBFS threshold.

    Args:
        segment (AudioSegment): Audio segment to filter.
        min_dbfs (float): Minimum dBFS threshold.

    Returns:
        list: List of valid 1-second AudioSegments.
    """
    valid_seconds = []
    total_seconds = len(segment) // 1000  # Number of full seconds

    for i in range(total_seconds):
        second = segment[i*1000:(i+1)*1000]
        try:
            dbfs = second.dBFS
        except:
            dbfs = -float('inf')  # Total silence

        if dbfs > min_dbfs:
            valid_seconds.append(second)
            print(f"Second {i+1}: dBFS {dbfs:.2f} exceeds threshold of {min_dbfs} dBFS")
        else:
            print(f"Second {i+1}: dBFS {dbfs:.2f} below threshold of {min_dbfs} dBFS")

    return valid_seconds

def normalize_audio_segment(audio_segment, target_dBFS=-20.0):
    """
    Normalizes an audio segment to the target dBFS level.

    Args:
        audio_segment (AudioSegment): Audio segment to normalize.
        target_dBFS (float): Target dBFS level.

    Returns:
        AudioSegment: Normalized audio segment.
    """
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def apply_compressor(audio_segment, threshold=-20.0, ratio=4.0):
    """
    Applies dynamic range compression to an audio segment using FFmpeg.

    Args:
        audio_segment (AudioSegment): Audio segment to compress.
        threshold (float): Threshold for compression in dBFS.
        ratio (float): Compression ratio.

    Returns:
        AudioSegment: Compressed audio segment.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
        audio_segment.export(temp_input_file.name, format="wav")
        temp_input_path = temp_input_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_file:
        temp_output_path = temp_output_file.name

    # FFmpeg command for dynamic range compression
    command = [
        "ffmpeg",
        "-y",
        "-i", temp_input_path,
        "-af", f"acompressor=threshold={threshold}dB:ratio={ratio}:attack=5:release=50",
        temp_output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error executing ffmpeg for compression: {result.stderr.decode()}")
        compressed_audio = audio_segment
    else:
        try:
            compressed_audio = AudioSegment.from_file(temp_output_path, format="wav")
        except Exception as e:
            print(f"Error loading compressed audio: {e}")
            compressed_audio = audio_segment

    # Clean up temporary files
    os.remove(temp_input_path)
    os.remove(temp_output_path)

    return compressed_audio

def generate_class_audio(class_name, metadata_filepath, class_dir, min_dbfs=-60, csv_records=None, final_audio_start_time=0.0):
    """
    Generates a 2-minute audio file for a specific class by processing YouTube videos.

    Args:
        class_name (str): Name of the class.
        metadata_filepath (str): Path to the CSV file with video metadata.
        class_dir (str): Directory where the class audio will be saved.
        min_dbfs (float): Minimum dBFS threshold for valid audio.
        csv_records (list, optional): List to append CSV records.
        final_audio_start_time (float, optional): Current start time in the final audio.

    Returns:
        tuple: (Success status, updated final_audio_start_time)
    """
    print(f"\nProcessing audio for class: {class_name}")
    final_audio = AudioSegment.empty()
    total_duration = 0  # in milliseconds
    target_duration = 120000  # 2 minutes in milliseconds

    # Read metadata from CSV
    try:
        with open(metadata_filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            metadata_list = list(reader)
        print(f"Metadata loaded successfully from '{metadata_filepath}'.")
    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_filepath}' not found.")
        return False, final_audio_start_time
    except Exception as e:
        print(f"Error loading metadata file '{metadata_filepath}': {e}")
        return False, final_audio_start_time

    if not metadata_list:
        print(f"No valid metadata entries found for class '{class_name}'. Skipping audio generation.")
        return False, final_audio_start_time

    # Initialize a pointer to reuse videos if necessary
    video_pointer = 0
    total_videos = len(metadata_list)

    while total_duration < target_duration:
        metadata = metadata_list[video_pointer]
        yt_url = metadata['video_url']
        start = float(metadata['start_time'])
        end = float(metadata['end_time'])
        output_filename = yt_url.split('v=')[-1]

        if download_youtube_audio(yt_url, output_filename, class_dir):
            file_path = os.path.join(class_dir, f"{output_filename}.mp3")
            try:
                audio = AudioSegment.from_file(file_path)
                segment = audio[start * 1000:end * 1000]
                print(f"Extracted segment: {start}s to {end}s, duration: {len(segment) / 1000:.2f} seconds")

                # Extract valid seconds based on dBFS
                valid_seconds = extract_valid_seconds(segment, min_dbfs)

                if valid_seconds:
                    for second in valid_seconds:
                        # Normalize each second before adding
                        normalized_second = normalize_audio_segment(second, target_dBFS=-20.0)
                        
                        if total_duration + len(normalized_second) > target_duration:
                            remaining_duration = target_duration - total_duration
                            if remaining_duration > 0:
                                normalized_second = normalized_second[:remaining_duration]
                                final_audio += normalized_second
                                
                                # Record the sample in CSV
                                if csv_records is not None:
                                    class_or_combination = sanitize_class_name(class_name)
                                    original_labels = metadata['original_labels']
                                    final_audio_end_time = final_audio_start_time + (len(normalized_second) / 1000)
                                    youtube_link = f"https://youtu.be/{output_filename}?t={int(start)}"
                                    csv_record = {
                                        'class_name': class_or_combination,
                                        'original_labels': original_labels,
                                        'final_audio_time': f"{final_audio_start_time:.1f} to {final_audio_end_time:.1f}",
                                        'youtube_link_1': youtube_link,
                                        'youtube_sample_time_1': f"{start:.1f} to {end:.1f}",
                                        'youtube_link_2': "",
                                        'youtube_sample_time_2': ""
                                    }
                                    csv_records.append(csv_record)
                                total_duration += len(normalized_second)
                                final_audio_start_time += len(normalized_second) / 1000
                                print(f"Current total duration: {total_duration / 1000:.2f} seconds")
                            break
                        final_audio += normalized_second
                        # Record the sample in CSV
                        if csv_records is not None:
                            class_or_combination = sanitize_class_name(class_name)
                            original_labels = metadata['original_labels']
                            final_audio_end_time = final_audio_start_time + (len(normalized_second) / 1000)
                            youtube_link = f"https://youtu.be/{output_filename}?t={int(start)}"
                            csv_record = {
                                'class_name': class_or_combination,
                                'original_labels': original_labels,
                                'final_audio_time': f"{final_audio_start_time:.1f} to {final_audio_end_time:.1f}",
                                'youtube_link_1': youtube_link,
                                'youtube_sample_time_1': f"{start:.1f} to {end:.1f}",
                                'youtube_link_2': "",
                                'youtube_sample_time_2': ""
                            }
                            csv_records.append(csv_record)
                        total_duration += len(normalized_second)
                        final_audio_start_time += len(normalized_second) / 1000
                        print(f"Current total duration: {total_duration / 1000:.2f} seconds")

                    if total_duration >= target_duration:
                        print("Reached target duration of 2 minutes for this class.")
                        os.remove(file_path)
                        break
                else:
                    print("Segment discarded due to low energy or silence.")

            except Exception as e:
                print(f"Error processing audio segment from '{file_path}': {e}")
            finally:
                # Remove the downloaded audio file to save space
                try:
                    os.remove(file_path)
                    print(f"File '{file_path}' removed.")
                except Exception as e:
                    print(f"Error removing file '{file_path}': {e}")

        else:
            print(f"Failed to download audio from '{yt_url}'. Skipping this video.")

        # Move to the next video
        video_pointer += 1
        if video_pointer >= total_videos:
            video_pointer = 0  # Restart from the beginning

            # If all videos have been traversed and target not met, warn the user
            if video_pointer == 0:
                print(f"Warning: Not enough valid audio segments for class '{class_name}'. Reusing videos from the start.")

    # Check if target duration was met
    if total_duration < target_duration:
        print(f"\nWarning: Class '{class_name}' did not reach 2 minutes. Current duration: {total_duration / 1000:.2f} seconds.")
        silence_needed = target_duration - total_duration
        final_audio += AudioSegment.silent(duration=silence_needed)
        print(f"Added {silence_needed / 1000:.2f} seconds of silence to complete 2 minutes.")

    else:
        print(f"\nClass '{class_name}' processed successfully with a duration of 2 minutes.")

    # Normalize the final audio
    normalized_final_audio = normalize_audio_segment(final_audio, target_dBFS=-20.0)
    if normalized_final_audio:
        # Apply dynamic range compression
        compressed_audio = apply_compressor(normalized_final_audio, threshold=-20.0, ratio=4.0)
        
        # Export the class-specific audio
        sanitized_name = sanitize_class_name(class_name)
        output_audio_path = os.path.join(class_dir, f"{sanitized_name}.wav")
        try:
            compressed_audio.export(output_audio_path, format="wav")
            print(f"Audio file for class '{class_name}' exported successfully as '{output_audio_path}'.")
            return True, final_audio_start_time
        except Exception as e:
            print(f"Error exporting audio file '{output_audio_path}': {e}")
            return False, final_audio_start_time
    else:
        print(f"Error normalizing audio for class '{class_name}'.")
        return False, final_audio_start_time

def generate_combination_audio(combination, root_dir=".", csv_records=None, final_audio_start_time=0.0):
    """
    Generates a 2-minute overlapped audio file for a given combination of classes.

    Args:
        combination (tuple): Tuple containing two class names.
        root_dir (str, optional): Root directory where class folders are located.
        csv_records (list, optional): List to append CSV records.
        final_audio_start_time (float, optional): Current start time in the final audio.

    Returns:
        tuple: (Success status, updated final_audio_start_time)
    """
    class1, class2 = combination
    sanitized_class1 = sanitize_class_name(class1)
    sanitized_class2 = sanitize_class_name(class2)
    class1_audio_path = os.path.join(root_dir, sanitized_class1, f"{sanitized_class1}.wav")
    class2_audio_path = os.path.join(root_dir, sanitized_class2, f"{sanitized_class2}.wav")

    if not os.path.exists(class1_audio_path):
        print(f"Error: Audio file for class '{class1}' not found at '{class1_audio_path}'.")
        return False, final_audio_start_time
    if not os.path.exists(class2_audio_path):
        print(f"Error: Audio file for class '{class2}' not found at '{class2_audio_path}'.")
        return False, final_audio_start_time

    try:
        audio1 = AudioSegment.from_file(class1_audio_path)
        audio2 = AudioSegment.from_file(class2_audio_path)

        # Normalize each audio based on dBFS
        target_dBFS = -20.0
        audio1 = normalize_audio_segment(audio1, target_dBFS=target_dBFS)
        audio2 = normalize_audio_segment(audio2, target_dBFS=target_dBFS)

        # Reduce volume to prevent clipping
        audio1 = audio1.apply_gain(-3)
        audio2 = audio2.apply_gain(-3)

        # Overlay the two audios without crossfade
        combined = audio1.overlay(audio2)

        # Ensure exact duration of 2 minutes
        if len(combined) > 120000:
            combined = combined[:120000]
        elif len(combined) < 120000:
            combined += AudioSegment.silent(duration=120000 - len(combined))

        # Normalize the combined audio
        normalized_combined = normalize_audio_segment(combined, target_dBFS=-18.0)
        if normalized_combined:
            combination_name = f"{sanitized_class1}_{sanitized_class2}"
            combination_dir = os.path.join(root_dir, combination_name)
            os.makedirs(combination_dir, exist_ok=True)
            output_audio_path = os.path.join(combination_dir, f"{combination_name}.wav")
            
            normalized_combined.export(
                output_audio_path,
                format="wav",
                parameters=["-ar", "44100", "-sample_fmt", "s16", "-dither_method", "triangular"]
            )
            
            print(f"Combination audio '{combination_name}.wav' exported successfully.")

            # Record the combination in CSV
            if csv_records is not None:
                # Original labels are combined from both classes
                labels1 = equivalence_labels.get(sanitized_class1, [sanitized_class1])
                labels2 = equivalence_labels.get(sanitized_class2, [sanitized_class2])
                original_labels = ";".join(labels1 + labels2)

                # For combination audios, assuming they are generic overlays, set 'N/A' for links
                csv_record = {
                    'class_name': combination_name,
                    'original_labels': original_labels,
                    'final_audio_time': f"{final_audio_start_time:.1f} to {final_audio_start_time + 120.0:.1f}",
                    'youtube_link_1': "N/A",
                    'youtube_sample_time_1': "N/A",
                    'youtube_link_2': "N/A",
                    'youtube_sample_time_2': "N/A"
                }
                csv_records.append(csv_record)
            
            final_audio_start_time += 120.0
            return True, final_audio_start_time
        else:
            print(f"Error normalizing combination audio for '{class1}' and '{class2}'.")
            return False, final_audio_start_time
    except Exception as e:
        print(f"Error generating combination audio for '{class1}' and '{class2}': {e}")
        return False, final_audio_start_time

def concatenate_combination_audios(combinations, root_dir=".", combined_audio_name="combined_audio.wav", csv_records=None, final_audio_start_time=0.0):
    """
    Concatenates all combination audio files into a single audio segment.

    Args:
        combinations (list): List of class combinations (tuples).
        root_dir (str, optional): Root directory where combination folders are located.
        combined_audio_name (str, optional): Name of the combined audio file.
        csv_records (list, optional): List to append CSV records.
        final_audio_start_time (float, optional): Current start time in the final audio.

    Returns:
        tuple: (Combined AudioSegment, updated final_audio_start_time)
    """
    combined_mix = AudioSegment.empty()
    target_dBFS = -20.0

    print("\nConcatenating combination audios...")
    for combination in combinations:
        success, final_audio_start_time = generate_combination_audio(
            combination, 
            root_dir=root_dir, 
            csv_records=csv_records, 
            final_audio_start_time=final_audio_start_time
        )
        if not success:
            print(f"Error generating combination audio for {combination}.")

    return combined_mix, final_audio_start_time

def concatenate_final_audio(classes, class_combinations, root_dir=".", final_audio_name="final_audio.wav"):
    """
    Concatenates class audios and combination audios into a final audio file and generates a CSV report.

    Args:
        classes (list): List of class names.
        class_combinations (list): List of class combinations (tuples).
        root_dir (str, optional): Root directory where audio folders are located.
        final_audio_name (str, optional): Name of the final audio file.

    Returns:
        None
    """
    final_mix = AudioSegment.empty()
    csv_records = []
    final_audio_start_time = 0.0  # Start time in the final audio

    # Define target dBFS for normalization
    target_dBFS = -20.0

    # Concatenate class audios and collect per-sample records
    for class_name in classes:
        sanitized_name = sanitize_class_name(class_name)
        class_audio_path = os.path.join(root_dir, sanitized_name, f"{sanitized_name}.wav")
        if os.path.exists(class_audio_path):
            try:
                class_audio = AudioSegment.from_file(class_audio_path)
                # Normalize based on dBFS
                class_audio = normalize_audio_segment(class_audio, target_dBFS=target_dBFS)
                final_mix += class_audio
                print(f"Class '{class_name}' added to final mix with normalized volume.")
                
                # Assuming per-sample records are already collected during class audio generation
                # Therefore, no need to add separate records here

                # Update final_audio_start_time
                final_audio_start_time += len(class_audio) / 1000  # in seconds
            except Exception as e:
                print(f"Error loading or normalizing class audio '{class_audio_path}': {e}")
        else:
            print(f"Warning: Audio file for class '{class_name}' not found at '{class_audio_path}'. Skipping.")

    print("\nClass audios concatenated and normalized based on dBFS.")

    # Generate and process combination audios
    print("\nGenerating combination audios...")
    for combination in class_combinations:
        success, final_audio_start_time = generate_combination_audio(
            combination, 
            root_dir=root_dir, 
            csv_records=csv_records, 
            final_audio_start_time=final_audio_start_time
        )
        if not success:
            print(f"Error generating combination audio for {combination}.")

    # Concatenate combination audios into final_mix
    print("\nConcatenating combination audios...")
    for combination in class_combinations:
        combination_name = f"{sanitize_class_name(combination[0])}_{sanitize_class_name(combination[1])}"
        combination_audio_path = os.path.join(root_dir, combination_name, f"{combination_name}.wav")
        if os.path.exists(combination_audio_path):
            try:
                combination_audio = AudioSegment.from_file(combination_audio_path)
                final_mix += combination_audio
                print(f"Combination audio '{combination_name}' added to final mix.")
                final_audio_start_time += len(combination_audio) / 1000  # in seconds
            except Exception as e:
                print(f"Error loading combination audio '{combination_audio_path}': {e}")
        else:
            print(f"Warning: Combination audio '{combination_audio_path}' not found. Skipping.")

    print("\nCombination audios concatenated and normalized based on dBFS.")

    # Verify total duration (should be 60 minutes: 30 for classes + 30 for combinations)
    expected_duration = (120000 * len(classes)) + (120000 * len(class_combinations))
    actual_duration = len(final_mix)
    if actual_duration != expected_duration:
        print(f"\nWarning: Final audio duration is {actual_duration / 1000 / 60:.2f} minutes, expected {expected_duration / 1000 / 60:.2f} minutes.")
        # Adjust duration by adding silence or truncating
        if actual_duration < expected_duration:
            silence_needed = expected_duration - actual_duration
            final_mix += AudioSegment.silent(duration=silence_needed)
            print(f"Added {silence_needed / 1000 / 60:.2f} minutes of silence to complete the final audio.")
        elif actual_duration > expected_duration:
            final_mix = final_mix[:expected_duration]
            print(f"Final audio truncated to {expected_duration / 1000 / 60:.2f} minutes.")
    else:
        print("\nFinal audio has the expected duration of 60 minutes.")

    # Normalize the final mix
    normalized_final_mix = normalize_audio_segment(final_mix, target_dBFS=target_dBFS)
    if normalized_final_mix:
        # Apply dynamic range compression
        compressed_final_mix = apply_compressor(normalized_final_mix, threshold=-20.0, ratio=4.0)
        
        # Export the final concatenated audio
        output_final_path = os.path.join(root_dir, final_audio_name)
        try:
            compressed_final_mix.export(output_final_path, format="wav")
            print(f"\nFinal audio exported successfully as '{output_final_path}'.")
        except Exception as e:
            print(f"Error exporting final audio '{output_final_path}': {e}")
    else:
        print("\nError normalizing the final audio mix.")

def generate_final_csv(root_dir, classes, class_combinations, final_csv_path):
    """
    Generates a comprehensive CSV mapping each audio segment in the final audio file
    to its originating YouTube video(s) and corresponding sample times.
    
    Args:
        root_dir (str): Root directory containing class folders and their metadata CSVs.
        classes (list): List of class names.
        class_combinations (list): List of class combinations for overlaying audios.
        final_csv_path (str): Path to save the final CSV file.
    """
    import pandas as pd
    import csv
    import os
    
    # Initialize containers for samples
    final_csv_rows = []
    current_time = 0.0  # Track position in final audio
    
    def sanitize_class_name(class_name):
        """Helper function to sanitize class names consistently"""
        return class_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('/', '_').lower()
    
    # Helper function to process a single class's metadata
    def process_class_metadata(class_name, start_time):
        class_samples = []
        sanitized_name = sanitize_class_name(class_name)
        metadata_path = os.path.join(root_dir, sanitized_name, f"{sanitized_name}_metadata.csv")
        
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found for class {class_name}")
            return class_samples, start_time
        
        try:
            df = pd.read_csv(metadata_path)
            # Process each 5-second segment that was actually used in the final audio
            segments_needed = 24  # 120 seconds / 5 seconds per segment
            
            for i in range(min(segments_needed, len(df))):
                row = df.iloc[i]
                sample = {
                    'class_name': sanitized_name,
                    'original_labels': row['original_labels'],
                    'final_audio_time': f"{start_time:.1f} to {start_time + 5.0:.1f}",
                    'youtube_link_1': row['video_url'],
                    'youtube_sample_time_1': f"{float(row['start_time']):.1f} to {float(row['end_time']):.1f}",
                    'youtube_link_2': "N/A",
                    'youtube_sample_time_2': "N/A"
                }
                class_samples.append(sample)
                start_time += 5.0
                
        except Exception as e:
            print(f"Error processing metadata for class {class_name}: {e}")
            
        return class_samples, start_time

    # Process individual classes (first 30 minutes)
    for class_name in classes:
        samples, current_time = process_class_metadata(class_name, current_time)
        final_csv_rows.extend(samples)
    
    # Process combinations (next 30 minutes)
    for combo in class_combinations:
        class1, class2 = combo
        sanitized_class1 = sanitize_class_name(class1)
        sanitized_class2 = sanitize_class_name(class2)
        sanitized_combo = f"{sanitized_class1}_{sanitized_class2}"
        
        # Get metadata for both classes
        class1_path = os.path.join(root_dir, sanitized_class1, f"{sanitized_class1}_metadata.csv")
        class2_path = os.path.join(root_dir, sanitized_class2, f"{sanitized_class2}_metadata.csv")
        
        try:
            # Read metadata for both classes
            class1_metadata = pd.read_csv(class1_path)
            class2_metadata = pd.read_csv(class2_path)
            
            # Process 24 segments (120 seconds) for this combination
            for i in range(24):
                # Get corresponding rows from both classes
                row1 = class1_metadata.iloc[i % len(class1_metadata)]
                row2 = class2_metadata.iloc[i % len(class2_metadata)]
                
                # Combine labels from both classes
                combined_labels = f"{row1['original_labels']};{row2['original_labels']}"
                
                sample = {
                    'class_name': sanitized_combo,
                    'original_labels': combined_labels,
                    'final_audio_time': f"{current_time:.1f} to {current_time + 5.0:.1f}",
                    'youtube_link_1': row1['video_url'],
                    'youtube_sample_time_1': f"{float(row1['start_time']):.1f} to {float(row1['end_time']):.1f}",
                    'youtube_link_2': row2['video_url'],
                    'youtube_sample_time_2': f"{float(row2['start_time']):.1f} to {float(row2['end_time']):.1f}"
                }
                final_csv_rows.append(sample)
                current_time += 5.0
                
        except Exception as e:
            print(f"Error processing combination {sanitized_combo}: {str(e)}")
            # Add placeholder row for error cases
            for i in range(24):
                sample = {
                    'class_name': sanitized_combo,
                    'original_labels': f"Error processing {class1} + {class2}",
                    'final_audio_time': f"{current_time:.1f} to {current_time + 5.0:.1f}",
                    'youtube_link_1': "Error",
                    'youtube_sample_time_1': "Error",
                    'youtube_link_2': "Error",
                    'youtube_sample_time_2': "Error"
                }
                final_csv_rows.append(sample)
                current_time += 5.0
    
    # Write to CSV
    fieldnames = [
        'class_name',
        'original_labels',
        'final_audio_time',
        'youtube_link_1',
        'youtube_sample_time_1',
        'youtube_link_2',
        'youtube_sample_time_2'
    ]
    
    try:
        with open(final_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_csv_rows)
        print(f"Final CSV file generated successfully at '{final_csv_path}'")
    except Exception as e:
        print(f"Error writing final CSV: {e}")

def main():
    """
    Main function to process classes, generate combination audios, and create the final audio.
    """
    parser = argparse.ArgumentParser(description="Generate and concatenate audio segments for classes and their combinations.")
    parser.add_argument(
        '--generate_class_audios',
        action='store_true',
        help='Generate only audio files for each class without creating the final audio.'
    )
    parser.add_argument(
        '--generate_final_audio',
        action='store_true',
        help='Generate only the final_audio.wav from existing class and combination audios without downloading/generating class audios.'
    )
    args = parser.parse_args()

    classes = CLASSES
    root_dir = "."  # Current directory; modify if necessary
    htmls_dir = "HTMLs"  # Directory containing HTML files
    cookies_path = None  # Path to the cookies file if necessary

    csv_records = []

    if not args.generate_final_audio:
        # Generate audios for classes only if not omitted
        for class_name in classes:
            sanitized_name = sanitize_class_name(class_name)
            class_dir = os.path.join(root_dir, sanitized_name)
            setup_directory(class_dir)
            html_filename = f"{sanitized_name}.html"
            html_filepath = os.path.join(htmls_dir, html_filename)
            metadata_filepath = os.path.join(class_dir, f"{sanitized_name}_metadata.csv")
            process_class_metadata(class_name, html_filepath, class_dir)
            # Generate audio based on metadata
            success, _ = generate_class_audio(class_name, metadata_filepath, class_dir, csv_records=csv_records)
            if not success:
                print(f"Error processing audio for class '{class_name}'.")

    if not args.generate_class_audios:
        # Concatenate class and combination audios into final_audio.wav
        concatenate_final_audio(classes, class_combinations, root_dir=root_dir, final_audio_name="final_audio.wav")

    final_csv = os.path.join(root_dir, "audio_samples_metadata.csv")
    generate_final_csv(root_dir, classes, class_combinations, final_csv)

if __name__ == "__main__":
    main()
