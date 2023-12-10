from moviepy.editor import VideoFileClip, TextClip,CompositeVideoClip
from moviepy.editor import concatenate_videoclips, ImageClip
from moviepy.editor import AudioFileClip, CompositeAudioClip, clips_array

import cv2
import pandas as pd
import requests
import numpy as np
import subprocess
import os

from tqdm import tqdm
import imutils
import string

from pexelsapi.pexels import Pexels
from skimage.metrics import structural_similarity as ssim
from openai import OpenAI
from moviepy.editor import VideoFileClip
import spacy

from moviepy.video.io.VideoFileClip import VideoFileClip

pexels_api_key = "QfGj5czSqWpkA3F27J8V9tTw5h7Eo50sZ6rstEUJt7bbbIIQZt4Th0wq"
CHAT_PICTURE_PROMPT = "I'll send you a list of nouns phrases. Which one of them is the most visually appealing in your mind and would fit well as a picture/video in a clip? You're answer should be **JUST** the noun.\n**NO ADDITIONAL WORDS!**"
PICTURE_GPT_MODEL = 'gpt-3.5-turbo'
chat_gpt_picture_api = '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'

cut_buffer = 0.15
colors = ['white', 'white','SpringGreen']
colors_back = ['DarkGreen','DarkGreen','black']
censor = {"fuck" : "f*ck", "shit" : "sh*t", "hore" : "h*re", "fucking" : "f*cking", "shiting" : "sh*ting", "sex" : "s*x"}

prototxt = 'deploy.prototxt'
model_face = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model_face)


def text_clip(vid, text: str, duration: int, start_time: int = 0):
    """Return a description string on the bottom-left of the video

    Args:
                text (str): Text to show
                duration (int): Duration of clip
               start_time (int, optional): Time in video to start at (in seconds). Defaults to 0.
    Returns:
                moviepy.editor.TextClip: A instance of a TextClip
    """
    ind = np.random.randint(0,3)
    color = colors[ind]
    if text in censor.keys():
      text = censor[text]
    return (TextClip(text.upper(), font="Lucida-Sans-Demibold-Roman", fontsize=70, color= color, stroke_color = colors_back[ind], stroke_width = 3, method = 'caption')
            .set_duration(duration).set_position('center')
            .set_start(start_time))

def add_subs_video(df, vid, start_times):
    # this code used to work with a general df for the whole video, now we have personlized df per video, more simple

    final_neg_vid = vid
    st = start_times

    # txt_clips_pos = [text_clip(pos_vid, new_subs_pos[i][0], new_subs_pos[i][2], new_subs_pos[i][1] - df.iloc[ind_pos]['start']) for i in range(len(new_subs_pos))]
    df = df[df['start_times'] >= st]

    txt_clips_neg = [text_clip(final_neg_vid, df.iloc[i]['text'], df.iloc[i]['end_times'] - df.iloc[i]['start_times'], df.iloc[i]['start_times'] - st) for i in range(len(df))]
    audio = final_neg_vid.audio
    final_neg_vid = CompositeVideoClip([final_neg_vid] + txt_clips_neg, use_bgclip=True)
    final_neg_vid = final_neg_vid.set_audio(audio)

    # THE PART THAT CUTS NONE SPOKEN TIME.,,
    # I DONT KNOW ...

    # added_min = 0
    # for k in range(len(df) - 1):
    #     if df.iloc[k]['end'] + cut_buffer < df.iloc[k+1]['end']:
    #
    #         final_neg_vid = final_neg_vid.cutout(new_subs_neg[k][2] + new_subs_neg[k][1]-added_min  - df.iloc[ind_neg]['start'] - st, new_subs_neg[k+1][1]-added_min  - df.iloc[ind_neg]['start'] - st - cut_buffer)
    #         added_min += new_subs_neg[k][2] + new_subs_neg[k][1] - new_subs_neg[k+1][1] - cut_buffer
    #
    #
    #     ret.append(final_neg_vid)

    return final_neg_vid

def cut_faces(neg_vid, stock_video): # stock_video in the format: (VIdeoFileClip object, query, final_start_time) or None if there is no need to add it
  boxes = []
  confs = []
  # ADD fps = X to run faster
  frames = [cv2.cvtColor(frame.astype('uint8'),cv2.COLOR_RGB2BGR) for frame in list(neg_vid.iter_frames())]
  dur = neg_vid.duration


  times = np.linspace(0, dur, len(frames), endpoint=False)

  # call for recognize faces model
  for ind_0 in tqdm(list(range(len(times)))):
    frame = frames[ind_0]

    # resize it to have a maximum width of 400 pixels
    image = imutils.resize(frame, width=400)
    (h, w) = image.shape[:2]

    # resize it to have a maximum width of 400 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    mx_confidence = 0
    boxes.append([])
    confs.append([])

    for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the prediction
      confidence = detections[0, 0, i, 2]
      mx_confidence = max(mx_confidence, confidence)

      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      if confidence > 0.4:
        boxes[-1].append((startX, endX, startY, endY))
        confs[-1].append(confidence)


  # decide where to do the cuts
  cuts_times = []
  cuts_poses = []
  last_xy = np.array([[-500,-500]], dtype='float64')
  last_cut = np.array([[-500,-500]], dtype='float64')
  FIXED_NUMBER = 10
  FAR_NUMBER = 30

  for ind, bs in enumerate(boxes):
    if len(bs) > 0:
        changed = False

        bs_sorted = [x for _, x in sorted(zip(confs[ind], bs), key=lambda pair: pair[0])]
        xys_sorted = np.array([[(b[1] + b[0]) / 2, (b[3] + b[2]) / 2] for b in bs_sorted])
        fixed = [np.min(np.linalg.norm(xys_sorted - last_xy_i, axis = 1)) <= FIXED_NUMBER for last_xy_i in last_xy]

        if sum(fixed) == 0:
          last_xy = np.array([[-500,-500]], dtype='float64')
          last_cut = np.array([[-500,-500]], dtype='float64')

        for i in range(len(fixed)):
            if fixed[i]:
                best_fit_ind = np.argmin(np.linalg.norm(xys_sorted - last_xy[i], axis = 1))
                last_xy[i] = xys_sorted[best_fit_ind]

        for xys in xys_sorted:
            break_loop = False
            for i in range(len(last_xy)):
                if (not fixed[i] and len(xys_sorted) > i) or (np.linalg.norm(last_xy[i] - xys) == 0):
                    last_xy[i] = np.array(xys)
                    changed = True
                    break_loop = True

            if (not break_loop) and (len(last_xy) == 1):
                last_xy = np.array(list(last_xy) + [np.array(xys)])
                changed = True
                fixed.append(True)

        if changed:
            if (len(last_cut) != len(last_xy)):
                last_cut = last_xy.copy()
                cuts_times.append(times[ind])
                cuts_poses.append([(last_xy[jk][0] / 400, last_xy[jk][1] / 400) for jk in range(len(last_xy))])

            elif not np.array([np.min(np.linalg.norm(last_cut - last_xy_i, axis=1)) <= FAR_NUMBER for last_xy_i in last_xy]).all():
                last_cut = last_xy.copy()
                cuts_times.append(times[ind])
                cuts_poses.append([(last_xy[jk][0] / 400, last_xy[jk][1] / 400) for jk in range(len(last_xy))])
  new_vids = []
  for i in range(len(cuts_times)):
      if i == len(cuts_times) - 1:
        if dur == cuts_times[i]:
            break
        sub_vid = neg_vid.subclip(cuts_times[i], dur)

      else:
        sub_vid = neg_vid.subclip(cuts_times[i], cuts_times[i+1])

      if len(cuts_poses[i]) == 1: #Easy fix
        sub_vid = sub_vid.resize(height=1280)
        sub_vid = sub_vid.crop(x1= min(max(0,(2275 * cuts_poses[i][0][0] - 360)), 2275 - 720), y1=0,x2=min(max((2275 * cuts_poses[i][0][0]) + 360,720), 2275),y2=1280)
        new_vids.append(sub_vid)

      elif len(cuts_poses[i]) == 2:
        sub_vid1 = sub_vid.resize(height=1280)
        sub_vid1 = sub_vid1.crop(x1=min(max(0, (2275 * cuts_poses[i][0][0] - 360)), 2275 - 720),
                                 y1=min(max(0, (1280 * cuts_poses[i][0][1] - int(1280 / 4))), 1280 - int(1280 / 2)),
                                 x2=min(max((2275 * cuts_poses[i][0][0]) + 360, 720), 2275),
                                 y2=min(max((1280 * cuts_poses[i][0][1]) + int(1280 / 4), int(1280 / 2)), 1280))

        sub_vid2 = sub_vid.resize(height=1280)
        sub_vid2 = sub_vid2.crop(x1=min(max(0, (2275 * cuts_poses[i][1][0] - 360)), 2275 - 720),
                                 y1=min(max(0, (1280 * cuts_poses[i][1][1] - int(1280 / 4))), 1280 - int(1280 / 2)),
                                 x2=min(max((2275 * cuts_poses[i][1][0]) + 360, 720), 2275),
                                 y2=min(max((1280 * cuts_poses[i][1][1]) + int(1280 / 4), int(1280 / 2)), 1280))

        if  cuts_poses[i][0][0] > cuts_poses[i][1][0]:
          new_vids.append(clips_array([[sub_vid1],
                            [sub_vid2]]))
        else:
          new_vids.append(clips_array([[sub_vid2],
                            [sub_vid1]]))
      else:
         print("Something is wrong.")
         print(f"The cut positions are: {cuts_poses[i]}")
  if not stock_video:
    final_neg_vid = concatenate_videoclips(new_vids)
  else:
    vid = concatenate_videoclips(new_vids)

    if stock_video[2] + stock_video[0].duration >= vid.duration:
      stock_video = (stock_video[0], stock_video[1], vid.duration - stock_video[0].duration - 3) # The 3 in the end is just random. Regardless we need to think where we need to put the stock videos
    pre_stock = vid.subclip(0, stock_video[2])
    audio = vid.subclip(stock_video[2], stock_video[2]+stock_video[0].duration).audio
    post_stock = vid.subclip(stock_video[2]+stock_video[0].duration)
    stock_vid = stock_video[0].set_audio(audio)
    final_neg_vid = concatenate_videoclips([pre_stock, stock_vid, post_stock])
  return final_neg_vid, cuts_times[0]


def is_substring(text1, text2):
  # Remove punctuation and convert to lowercase
  text1_cleaned = text1.translate(str.maketrans("", "", string.punctuation)).lower()
  text2_cleaned = text2.translate(str.maketrans("", "", string.punctuation)).lower()

  # Check if one text is a substring of the other
  answer = text1_cleaned in text2_cleaned or text2_cleaned in text1_cleaned
  return answer


def get_relevant_video(text, temp_folder, res_per_page=50):
    query = chat_gpt_noun_request(text)
    pexel = Pexels(pexels_api_key)
    video_found = False
    page_num = 1
    max_page_num = 10 # Random number I wrote. We actually need to check what happens if there are no more results
    input_video_path = f"{temp_folder}//{query}.mp4"

    while not video_found and page_num <= max_page_num:
      print(f"Currently searching for {query} on page number: {page_num} out of {max_page_num}")
      search_videos = pexel.search_videos(query=query, orientation='', size='', color='', locale='', page=page_num, per_page=res_per_page)
      for video in search_videos['videos']:
        if video['height']/video['width'] == 16/9:
          video_id = video['id']
          video_found = True
          break
      page_num += 1
    if not video_found:
      return None # Needs to be compatible later with the handling of the info from this function
    video_url = 'https://www.pexels.com/video/' + str(video_id) + '/download'
    r = requests.get(video_url)

    with open(input_video_path, 'wb') as outfile:
        outfile.write(r.content)
    try:
      video = VideoFileClip(input_video_path)
      # There is something wrong with the metadata of some of the videos being saved so this code recognises this, deletes the video and returns None
      if video.h > video.w:
        sub_clip = video.subclip(0, min(video.duration, 5))
        final = sub_clip.resize(height=1280, width=720)
        ret_val = (final, query)
      else:
        print(f"Metadata is wrong for {query} video")
        ret_val = None
    except Exception as e:
      print(f"Error processing video '{input_video_path}': {e}")
      ret_val = None
    return ret_val

def chat_gpt_noun_request(text):
  nlp = spacy.load("en_core_web_trf")
  doc = nlp(text)
  unique_chunks = {}

# Iterate through noun chunks and add them to the dictionary
  for chunk in doc.noun_chunks:
    unique_chunks[chunk.text] = True

  chat_request = CHAT_PICTURE_PROMPT + '\n' + '\n'.join(unique_chunks.keys())
  conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": chat_request}
        ]
  # Make a request to the ChatGPT API
  client = OpenAI(api_key=chat_gpt_picture_api)
  response = client.chat.completions.create(
      model=PICTURE_GPT_MODEL,
      messages=conversation
  )
  # Extract and display the assistant's reply
  assistant_reply = clean_str(response.choices[0].message.content)
  # Output
  return assistant_reply


def get_start_times(key_words_lst, dfs):
  target_words = [clean_str(key_words).split() if key_words else None for key_words in key_words_lst]
  start_times = [filter_rows(target_words[i], dfs[i]) if target_words[i] else None for i in range(len(target_words))]
  print(f"Start times are: {start_times}")
  return start_times

def filter_rows(target_words, df):
  for i in range(len(df) - len(target_words) + 1):
    # lst = df.iloc[i:i+len(target_words)-1]['text'].tolist()
    window_words = df.iloc[i:i+len(target_words)]['text'].tolist()
    window_words_clean = [clean_str(wrd) for wrd in window_words] # Sometimes empty at the end of the loop, not sure how or why that would be the case
    if window_words_clean == target_words:
      return df.iloc[i]['start']
  return None

def clean_str(s):
  # Returns string with no punctuation marks in all lower case
  return s.translate(str.maketrans('', '', string.punctuation)).lower()


if __name__ == "__main__":
  pass