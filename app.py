from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
import os
import threading
import datetime
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from dotenv import load_dotenv

from best_clips import BestClips

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv('DB_URI')
mongo_client = MongoClient(mongo_uri)
db = mongo_client['test']

app = Flask(__name__)

# Enable CORS with support for credentials and specific origins
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3001"}})

# Specify the origins that are allowed to connect for SocketIO
# Adjust the origins according to your environment. Use '*' to allow all origins (not recommended for production).
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3001", logger=True, engineio_logger=True)

google_cloud_key_file = os.getenv('GOOGLE_CLOUD_KEY_FILE')
if not google_cloud_key_file or not os.path.exists(google_cloud_key_file):
    raise ValueError("GOOGLE_CLOUD_KEY_FILE environment variable not set or file does not exist.")

print(f"GOOGLE_CLOUD_KEY_FILE: {google_cloud_key_file}")


def generate_signed_url(bucket_name, blob_name):
    try:
        storage_client = storage.Client.from_service_account_json(google_cloud_key_file)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            print(f"Blob {blob_name} does not exist in the bucket {bucket_name}.")
            return None

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=1500),
            method="GET"
        )
        return url
    except GoogleCloudError as e:
        print(f"Failed to generate signed URL for {blob_name}: {e}")
        return None

def set_upload_complete(userEmail, complete):
    try:
        # The upsert=True option is used to create a new document if one doesn't already exist
        result = db.users.update_one(
            {'email': userEmail},
            {'$set': {'upload_complete': complete}},
            upsert=True
        )
        print(f"Update result for {userEmail}: {result.matched_count} matched, {result.modified_count} modified, upserted_id: {result.upserted_id}")
    except Exception as e:
        print(f"An error occurred while setting upload_complete for {userEmail}: {e}")
        raise e  # Reraising the exception will help to identify if there is an issue with the database operation

def upload_to_gcloud(bucket, video_file_name, json_file_name, video_destination_blob_name, json_destination_blob_name, userEmail):
    if not userEmail:
        print("Error: User ID is None or empty.")
        return False
    
    user_prev_runs_path_video = f"{userEmail}/PreviousRuns/{video_destination_blob_name}"
    user_cur_run_path_video = f"{userEmail}/CurrentRun/{video_destination_blob_name}"
    user_prev_runs_path_json = f"{userEmail}/PreviousRuns/{json_destination_blob_name}"
    user_cur_run_path_json = f"{userEmail}/CurrentRun/{json_destination_blob_name}"

    if not os.path.isfile(video_file_name):
        print(f"The file {video_file_name} does not exist.")
        return False
    
    if not os.path.isfile(json_file_name):
        print(f"The file {json_file_name} does not exist.")
        return False

    try:
        blob_prev_video = bucket.blob(user_prev_runs_path_video)
        blob_prev_video.upload_from_filename(video_file_name)
        blob_prev_json = bucket.blob(user_prev_runs_path_json)
        blob_prev_json.upload_from_filename(json_file_name)
        blob_cur_video = bucket.blob(user_cur_run_path_video)
        blob_cur_video.upload_from_filename(video_file_name)
        blob_cur_json = bucket.blob(user_cur_run_path_json)
        blob_cur_json.upload_from_filename(json_file_name)

        print(f"File {video_file_name} and {json_file_name} uploaded to {user_cur_run_path_video}, {user_prev_runs_path_video} and {user_cur_run_path_json}, {user_prev_runs_path_json} respectively.")
        return True
    except Exception as e:
        print(f"Failed to upload {video_file_name} or {json_file_name}: {e}")
        return False


def process_youtube_video(link, userEmail):
    set_upload_complete(userEmail, False)  # Set the upload_complete flag to False at the start
    
    try:
        username = userEmail  # Use userEmail for the folder name

        # Pass save_folder_name to BestClips constructor
        best_clips = BestClips(link, username, use_gpt=True) # Change use_gpt to True if you're not debugging and want to see the best parts
        
        gcloud_bucket_name = "clipitshorts"
        # Delete all of the files in user_cur_run_path if they exist
        storage_client = storage.Client.from_service_account_json(google_cloud_key_file)
        bucket = storage_client.bucket(gcloud_bucket_name)
        blobs = bucket.list_blobs(prefix=f"{userEmail}/CurrentRun/")
        for blob in blobs:
            blob.delete()
        for i in range(len(best_clips.final_shorts)):
            video_file_path = os.path.join(best_clips.run_path, f"short_{str(i)}.mp4")
            json_file_path = os.path.join(best_clips.run_path, f"short_{str(i)}.json")
            gcloud_video_destination_name = f"{best_clips.date_time_str}__{os.path.basename(video_file_path)}"
            gcloud_json_destination_name = f"{best_clips.date_time_str}__{os.path.basename(json_file_path)}"
            upload_to_gcloud(bucket, video_file_path, json_file_path, gcloud_video_destination_name, gcloud_json_destination_name, userEmail)
        set_upload_complete(userEmail, True)
        
    except Exception as e:
        print(f"An error occurred in process_youtube_video: {e}")
        # Handle any other exceptions here


@app.route('/api/process-youtube-video', methods=['POST'])

def handle_youtube_video():
    data = request.json
    youtube_link = data.get('link')
    userEmail = data.get('userEmail')  # Extract user ID from the request
    print(f"Received userEmail: {userEmail}")  # Add this line for debugging


    if not youtube_link:
        return jsonify({'error': 'No YouTube link provided'}), 400
    if not userEmail:
        return jsonify({'error': 'No user ID provided'}), 400

    try:
        thread = threading.Thread(target=process_youtube_video, args=(youtube_link, userEmail))
        thread.start()
        return jsonify({'message': 'YouTube video processing started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_signed_urls_for_directory(bucket, directory):
    storage_client = storage.Client.from_service_account_json(google_cloud_key_file)
    blobs = storage_client.list_blobs(bucket, prefix=directory)
    signed_urls = []
    for blob in blobs:
        # Check if the blob name ends with '.mp4'
        if blob.name.lower().endswith('.mp4'):
            url = generate_signed_url(bucket, blob.name)
            if url:
                signed_urls.append(url)
    return signed_urls


@app.route('/api/signed-urls', methods=['GET'])
def get_signed_urls():
    try:
        email = request.headers.get('User-Email')
        if not email:
            return jsonify({'error': 'User email is required'}), 400

        directory = request.args.get('directory', default=f'{email}/CurrentRun/')
        bucket_name = 'clipitshorts'

        signed_urls = get_signed_urls_for_directory(bucket_name, directory)

        # Check if 'CurrentRun' is empty and fetch from 'undefined' if needed
        if not signed_urls and 'CurrentRun' in directory:
            directory = f'undefined/'
            signed_urls = get_signed_urls_for_directory(bucket_name, directory)

        return jsonify({'signedUrls': signed_urls})
    except GoogleCloudError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/api/user/payment-plan', methods=['GET'])
def get_user_payment_plan():
    user_email = request.args.get('email')

    if not user_email:
        return jsonify({'error': 'Email is required'}), 400

    user = db.users.find_one({'email': user_email})

    if not user:
        return jsonify({'error': 'User not found'}), 404

    payment_plan = user.get('paymentPlan', 'free')
    return jsonify({'paymentPlan': payment_plan})

if __name__ == '__main__':
    socketio.run(app, debug=True)
