from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from google.cloud import speech
import whisper
import pandas as pd
import os
import io
import json
import re
import requests
from download_youtube_data_utils import parse_chapters

from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN", "chromedriver")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())

from moviepy.editor import VideoFileClip, TextClip,CompositeVideoClip
from moviepy.editor import concatenate_videoclips, ImageClip
from moviepy.editor import AudioFileClip, CompositeAudioClip, clips_array

import matplotlib.pyplot as plt
import numpy as np
from pytube import YouTube
from scipy.signal import find_peaks
import os
import shutil
import pickle

DRIVER_PATH = "C://Users//along//VS Code//Shorts Project//chromium//chromedriver.exe"
api_key = 'AIzaSyDuCDLpv1XKwlv1ZLeG8WSkyEH2PwHRgkk'
WHISPER_MODEL = "medium.en"


class MyCustomException(Exception):
    pass
class YoutubeData:
    def __init__(self, link, name, n_vids=5, dest="downloaded_files/",buffer = 69):
        self.buffer = buffer
        self.name = name
        self.link = link
        self.dest = dest + name + "/"
        self.create_fold()
        self.n_vids = n_vids
        self.download_data()

    def create_fold(self):
        try:
            os.mkdir(self.dest)
        except:
            inp = input("This name exists, do you want to delete it? [Y/n]")
            if inp.lower() == 'y':
                shutil.rmtree(self.dest)
                os.mkdir(self.dest)
            else:
                raise MyCustomException("Proccses stoped.")

    def get_heatmap(self):
        link = self.link
        chrome_options = webdriver.ChromeOptions()
        service = Service()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.headless = True
        wd = webdriver.Chrome(service=service, options=chrome_options)
        wd.get(link)
        # Wait for the specific element to be present
        wait = WebDriverWait(wd, 30)  # wait for a maximum of 10 seconds
        els = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.ytp-heat-map-svg')))

      # If you want to get specific attributes or text from these elements, you can loop over them:
        # TODO: NEED TO JOIN THE RETS SOMEHOW
        rets = []
        for el in els:
            rets.append(str(el.get_attribute('outerHTML')))

        return rets

    def preProcessData(self):
        all_dpa = []
        for ret in self.raw_data:
            data = ret.split('class="ytp-heat-map-path" d=')[1].split('" fill="white">')[0]

            tripletsArray = data.split("C ")

            dataPointsArray = []

            for triplets in tripletsArray[1:]:
                if triplets != "":

                    pointsArray = triplets.split(" ")[:3]

                    for points in pointsArray:
                        p = points.split(",")

                        # print(p)
                        dataPointsArray.append([float(p[0]), float(p[1])])

            all_dpa.append(dataPointsArray)
        return all_dpa

    def plotCurve(self,all_points):
        all_xs = []
        all_ys = []

        for i, points in enumerate(all_points):
            x = np.array([((p[0] - 1) / (1000 - 1)) for p in points])
            y = np.array([-p[1] for p in points])
            inds = (x >= 0) & (x < 1)

            x = x[inds]
            y = y[inds]
            if i < len(all_points) - 1:
                x = x * (self.chapters[i+1]['timestamp'] - self.chapters[i]['timestamp'])+ self.chapters[i]['timestamp']
            else:
                if len(self.chapters) > 0:
                    x = x * (self.length - self.chapters[i]['timestamp'])+ self.chapters[i]['timestamp']
                else:
                    x = x * self.length

            all_xs += list(x)
            all_ys += list(y)

        inds = find_peaks(all_ys, distance=10)[0]
        inds2 = np.argpartition(np.array(all_ys)[inds], -self.n_vids)[-self.n_vids:]

        return all_xs,all_ys, np.array(all_xs)[inds][inds2] / self.length


    def download_audio_video(self):
        
        link = self.link
        destination = self.dest
        # url input from user
        yt = YouTube(link)

        # extract only audio
        video = yt.streams.filter(only_audio=True).first()

        # download the file
        out_file = video.download(output_path=destination)

        # save the file
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)

        youtube = YouTube(link)

        video = youtube.streams.get_highest_resolution()

        out_file = video.download(destination)

        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp4'
        os.rename(out_file, new_file)

        return new_file[:-4], yt.length

    def save_wanted_parts(self):
        interesting_points = self.interesting_points
        filename = self.filename + ".mp3"
        buffer = self.buffer
        audio = AudioFileClip(filename)
        length = audio.duration
        spans = []
        filenames = []
        for i,x in enumerate(interesting_points):

            min_t = int(max(x * length - buffer, 0) + max(x * length - length, 0))
            max_t = int(min(x * length + buffer, length) + max(- x * length + buffer, 0))
            new_aud = audio.subclip(min_t, max_t)
            spans.append((min_t, max_t))
            new_aud.write_audiofile(filename[:-4] + "_tmp" + str(i) + ".wav", fps = 44100,codec='pcm_s16le')
            filenames.append(filename[:-4] + "_tmp" + str(i) + ".wav")

        return spans, filenames

    def transcribe_file(self, filename):
        texts = []
        starts = []
        ends = []

        # Whisper STT
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(filename, word_timestamps=True)
        for segment in result['segments']:
            segment_word_data = segment['words']
            for word_data in segment_word_data:
                # print(f"Word:{word_data['word']}. Start: {word_data['start']}. End: {word_data['end']}")
                # if word_data['probability'] < 0.8:
                #         print(f"Accuracy of{word_data['word']} only {word_data['probability']}")
                texts.append(word_data['word'].lstrip())
                starts.append(word_data['start'])
                ends.append(word_data['end'])

        
        # # Google's Cloud Speech STT
        # client = speech.SpeechClient(client_options={"api_key": api_key})
        # file_name = filename

        # with io.open(file_name, "rb") as audio_file:
        #  content = audio_file.read()
        #  audio = speech.RecognitionAudio(content=content)

        # config = speech.RecognitionConfig(
        #  encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #  audio_channel_count = 2,
        #  language_code="en-US",
        #  enable_word_time_offsets=True,
        # )

        # operation = client.long_running_recognize(config=config, audio=audio)
        # print("Waiting for operation to complete...")
        # result = operation.result(timeout=90)
        # for result in result.results:
        #     alternative = result.alternatives[0]
        #     print(f"Transcript: {alternative.transcript}")
        #     print(f"Confidence: {alternative.confidence}")

        #     for word_info in alternative.words:
        #         word = word_info.word
        #         start_time = word_info.start_time
        #         end_time = word_info.end_time
        #         texts.append(word)
        #         starts.append(start_time)
        #         ends.append(end_time)

        df = pd.DataFrame(np.array([texts,starts,ends]).transpose(), columns = ["text","start","end"])
        return df

    def extract_chapters(self):
        # Load the YouTube video
        full_html = requests.get(self.link).text
        y = re.search(r'shortDescription":"', full_html)
        desc = ""
        count = y.start() + 19  # adding the length of the 'shortDescription":"
        while True:
            # get the letter at current index in text
            letter = full_html[count]
            if letter == "\"":
                if full_html[count - 1] == "\\":
                    # this is case where the letter before is a backslash, meaning it is not real end of description
                    desc += letter
                    count += 1
                else:
                    break
            else:
                desc += letter
                count += 1

        return parse_chapters(desc)

    def transcribe(self):
        dfs = []
        for file in self.filenames:
            print(file)
            df = self.transcribe_file(file)
            df.to_csv(file[:-4] + ".csv")
            dfs.append(df)
        return dfs
    
    def save_object(self):
        with open(self.dest + "object_data.pkl", 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def download_data(self):
        print("extracting the chapters")
        self.chapters = self.extract_chapters()
        print("downloading file")
        self.filename, self.length = self.download_audio_video()
        print("getting most replayed")
        self.raw_data = self.get_heatmap()
        all_points = self.preProcessData()
        self.x,self.y,self.interesting_points = self.plotCurve(all_points)
        # self.interesting_points = self.interesting_points / max(self.interesting_points)
        print(self.interesting_points)

        print("saving the intersting parts")
        self.spans, self.filenames = self.save_wanted_parts()
        print("saving the object")
        self.save_object()
        self.transcribe()


def load_youtube_info(name, dest="downloaded_files/"):
    fn = dest + name + "/object_data.pkl"
    with open(fn, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

if __name__ == "__main__":
    link = "https://www.youtube.com/watch?v=8EIx_aCmg2g"
    YoutubeData(link,"first_test")