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

from download_youtube_data import YoutubeData
from edit_youtube_video import EditedVideos

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

def upload_to_gcloud(bucket_name, source_file_name, destination_blob_name, userEmail):
    if not userEmail:
        print("Error: User ID is None or empty.")
        return False

    # Create a user-specific path in the Google Cloud Storage bucket
    user_specific_path = f"{userEmail}/{destination_blob_name}"

    print(f"Uploading to Google Cloud Storage: {user_specific_path}")

    if not os.path.isfile(source_file_name):
        print(f"The file {source_file_name} does not exist.")
        return False

    try:
        storage_client = storage.Client.from_service_account_json(google_cloud_key_file)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(user_specific_path)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {user_specific_path}.")
        return True
    except Exception as e:
        print(f"Failed to upload {source_file_name} to {user_specific_path}: {e}")
        return False



def process_youtube_video(link, userEmail):
    set_upload_complete(userEmail, False)  # Set the upload_complete flag to False at the start
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Change the save_folder_name to use the userEmail
    save_folder_name = userEmail  
    dest_folder = os.path.join(base_dir, save_folder_name)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    yt_data_obj = YoutubeData(link, save_folder_name, dest=dest_folder)
    yt_data_obj.save_object()

    edited_videos = EditedVideos(yt_data_obj, load_gpt=False)

    gcloud_bucket_name = "clipitshorts"
    for i in range(len(edited_videos.faced_subs_vids)):
        video_file_path = os.path.join(yt_data_obj.dest, "finalvideo" + "_" + str(i) + ".mp4")
        gcloud_destination_name = os.path.join(save_folder_name, os.path.basename(video_file_path))
        # Use the userEmail as the folder name in the upload function
        upload_to_gcloud(gcloud_bucket_name, video_file_path, gcloud_destination_name, userEmail)
    set_upload_complete(userEmail, True)

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


@app.route('/api/signed-urls', methods=['GET'])
def get_signed_urls():
    # Extract user email from the request arguments instead of using a hardcoded value
    email = request.args.get('email')
    print(f"get_signed_urls called with email: {email}")  # Additional logging for debugging
    if not email:
        return jsonify({'error': 'User email is required'}), 400

    bucket_name = 'clipitshorts'
    user = db.users.find_one({'email': email})
    if not user:
        # If the user is not found, assume no upload has started for this user
        directory_name = 'undefined/'
    else:
        # Use the directory based on the user's upload status
        directory_name = email + '/' if user.get('upload_complete', False) else 'undefined/'

    storage_client = storage.Client.from_service_account_json(google_cloud_key_file)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=directory_name))
    signed_urls = [generate_signed_url(bucket_name, blob.name) for blob in blobs if not blob.name.endswith('/')]

    return jsonify({'signedUrls': signed_urls})


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
