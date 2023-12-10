from pymongo import MongoClient

# Replace 'your_mongodb_uri' with your actual MongoDB URI.
mongo_uri = 'mongodb+srv://korren:Kokoman10@cluster0.hwmulir.mongodb.net/test?retryWrites=true&w=majority'

try:
    client = MongoClient(mongo_uri)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB is connected and accessible")
except Exception as e:
    print("An error occurred while trying to connect to MongoDB:", e)
