import re

from download_youtube_data import YoutubeData
from tqdm import tqdm
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.editor import concatenate_videoclips, ImageClip
from moviepy.editor import AudioFileClip, CompositeAudioClip

import imutils
import numpy as np
import pandas as pd
import cv2
import gc

import os
from openai import OpenAI
import pickle
import shutil
import time

from edit_youtube_video_utils import cut_faces, get_start_times, add_subs_video, get_relevant_video

SAVED_NAME = "first_test"
USING_WHISPER = True # Change if using Google Cloud's Speech API

# FIRST_INPUT = "I'm going to send you a text.\nI need you to summarize that text using only direct quotes from it.\nDo it with no fillers or additional words, just **DIRECT QUOTES**. If something doesn't make sense feel free to omit it.\nWhen the quotes are combined, I expect the result to be:\n- Simple\n- Interesting\n- To keep the original meaning\nKeep it short (no more than 100 words). Format the summary as a list of quotes from the original text, each one starting with a '-'.\nDon't use ellipsis under any circumstance!"
# FIRST_RES = "Of course! Please provide the text you'd like me to summarize using direct quotes, and I'll follow your instructions."

FIRST_RES = "Of course! Please provide the text you'd like me to summarize using direct quotes, and I'll follow your instructions."
FIRST_INPUT = "\n send me a full long list that numbers all the words in the original text I sent you"
# SECOND_INPUT = "Create a COHERENT summary of the MAIN IDEA OF THE TEXT using only **comprehensive** and **LONG** quotes from the text. WRITE the INDEX RANGE of each quotation on the LIST! Format: (x-y)"
# SECOND_INPUT = "This is a transcription of a section from a podcast.\nI want to make an interesting 1 minute clip using only quotations from the video. Choose the quotations for me. \nThe concatenation of them has to MAKE SENSE and BE COHERENT as a short video.\nYou don't have to include all of the subjects from the test. \nUse UP TO 5 quotes.\nwrite the EXACT INDEX RANGE of each quotation on the LIST."
SECOND_INPUT = "Create a COHERENT SUMMERY of the MAIN IDEA OF THE TEXT using only comprehensive LONG quotes from the text.\nWRITE the INDEX RANGE of each quotation on the LIST!"

TEST_FIRST_INP = "I'm going to send you a trascript of a podcast as a list of words with indices. I want to make an interesting 1 minute clip from the podcast.\nFor that, I want you to send me the indices of the quotes from the trsnscript, which I should include in my clip.\nLet's check that you understand the format of your response: 1. Alright,\n2. listen,\n3. I\n4. gotta\n5. tell\n6. you\n7. guys\n8. about\n9. this\n10. mind-blowing\n11. conversation\n12. I\n13. had\n14. the\n15. other\n16. day.\n17. I\n18. had\n\n19. this\n20. expert\n21. on\n22. the\n23. podcast,\n24. right?\n25. And\n26. we\n27. started\n28. talking\n29. about\n30. the\n31. moon\n32. landing.\n33. Now,\n34. I\n35. know,\n36. I\n37. know,\n38. we've\n39. all\n40. heard\n41. the\n42. stories,\n43. but\n44. this\n45. one,\n46. man,\n47. it's\n48. next\n49. level."
TEST_FIRST_RES = "(3-24), (26-33), (36-49)"
TEST_SECOND_INP = "Great! KEEPING THE SAME FORMAT IN YOUR RESPONSE AND ADDING NO MORE WORDS, here is the transcript:"

# BASIC_PROMPT = "delete the less relavent parts of the next text. Don't rewrite ANYTHING just delete parts. remain with 100 words\n"
openai_api_key = "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"
extra_time = 0 # Dont change if you dont fix usage in get time intervals in creating the ret_df



def get_seconds(t):
    ts = t.split('days')[1].split(':')
    ret = float(ts[0]) * 3600 + float(ts[1]) * 60 + float(ts[2])
    return ret

class EditedVideos:
    def __init__(self, youtube_data, load_gpt = False):
        self.youtube_data = youtube_data
        self.temp_folder = self.create_temp_folder()
        if not load_gpt:
            self.text_parts = []
            self.dfs = [pd.read_csv(fn[:-4] + ".csv") for fn in self.youtube_data.filenames]
            self.do_nlp()
            with open(self.youtube_data.dest + "after_gpt_object_data.pkl", 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

        else:
            with open(self.youtube_data.dest + "after_gpt_object_data.pkl", 'rb') as inp:
                obj = pickle.load(inp)
                self.text_parts = obj.text_parts
                self.dfs = obj.dfs
                self.time_intervals = obj.time_intervals
                # print(self.time_intervals)
        print("getting relevant stock video")
        self.stock_videos = self.relevant_videos()
        print("cutting vids")
        self.vids = self.cut_vids()
        print("cutting around faces vids")
        self.faced_vids, self.new_start_times = self.cuts_faces()
        print("adding subs")
        self.faced_subs_vids = self.add_subs()
        print("saving files!")
        self.save_vids()

        # Currently not working and need to delete the temp_folder library manually
        # self.del_temp_folder()

    def relevant_videos(self):
        return [get_relevant_video(text, self.temp_folder) for text in self.text_parts]
    
    def create_temp_folder(self):
          temp_folder_name = "temp_files"
          os.mkdir(temp_folder_name)
          return temp_folder_name

    def del_temp_folder(self):
        shutil.rmtree(self.temp_folder)

    def cuts_faces(self):
        faced_vids = []
        new_start_times = []
        for i in range(len(self.vids)):
            faced_vid, start_time = cut_faces(self.vids[i], self.stock_videos[i]) # neg time to cut time
            faced_vids.append(faced_vid)
            new_start_times.append(start_time)

        return faced_vids, new_start_times
    
    def add_subs(self):
        subed_faced_vids = []
        for i, vid in tqdm(enumerate(self.faced_vids)):
            # call to add subs 
            subed_faced_vids.append(add_subs_video(self.dfs[i], vid, start_times=self.new_start_times[i]))
        return subed_faced_vids # return the videos with subs
    
    def save_vids(self):
        for i in range(len(self.faced_subs_vids)):
            self.faced_subs_vids[i].write_videofile(self.youtube_data.dest + "finalvideo" + "_" + str(i) + ".mp4", fps=24, audio_codec='aac')

    def do_nlp(self):
        self.time_intervals = []
        for ind in range(len(self.dfs)):
            df, text_part = self.read_df(ind)
            self.dfs[ind] = df
            self.text_parts.append(text_part)
            time_intervals, df = self.get_summeraztion_times(ind)
            print(f"Time Intervals: {time_intervals}")
            if len(time_intervals) == 0:
                print(text_part)
            self.time_intervals.append(time_intervals)
            self.dfs[ind] = df

            # print(df['text'].to_string(index=True))

    def read_df(self, ind):
        df = self.dfs[ind]
        text_part = ' '.join(str(df['text']))
        if not USING_WHISPER:
            df['start'] = df['start'].apply(get_seconds)
            df['end'] = df['end'].apply(get_seconds) # This replaced the line of code beneath which is commented out. Still need to check that it maintains the same functionality and logic
        # df['end'] = df['end'].apply(lambda x : x.total_seconds)
        df['start_times'] = df['start']
        df['end_times'] = df['end']
        return df, text_part

    def call_chat_gpt(self, ind):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": TEST_FIRST_INP},
            {"role": "assistant", "content": TEST_FIRST_RES},
            {"role": "user", "content": TEST_SECOND_INP + "\n" + "\n".join(f"{index + 1}. {row['text']}" for index, row in self.dfs[ind].iterrows())},
        ]

        # print("\n".join(f"{index + 1}. {row['text']}" for index, row in self.dfs[ind].iterrows()))
        # Make API request with the conversation so far
        # conversation = [
        #     # {"role": "system", "content": "You are a helpful assistant."},
        #     # {"role": "user", "content": self.text_parts[ind] + FIRST_INPUT},
        #     {"role": "user", "content": "\n".join(f"{index + 1}. {row['text']}" for index, row in self.dfs[ind].iterrows())},
        #     {"role": "user", "content": SECOND_INPUT}
        # ]

        # Make a request to the ChatGPT API
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=conversation
        )

        # Extract and display the assistant's reply
        assistant_reply = response.choices[0].message.content
        # print(assistant_reply)
        # Output
        return assistant_reply

    def get_time_intervals(self, df, assistant_reply, ind, debugging=False):
        # lst = assistant_reply.replace("...", "").split('"') # added a different line of code to skip ver incomplete sentences that end in an ellipsis. If there's a set of answers which need this line then it needs to be reimplemented
        if not debugging:
            regg = "\d{1,3}\s*-\s*\d{1,3}"
            spans = list(dict.fromkeys(re.compile(regg).findall(assistant_reply.replace("Index",""))))
            # change indexing from 1 - ..., to 0 - ....
            all_locs = [(int(x.split('-')[0]) - 1,int(x.split('-')[1]) - int(x.split('-')[0])) for x in spans]

            # In case all_locs is empty, which happens when Chat_GPT returns an answer which isn't in the expected format,
            # no time intervals will be made and thus no video. In addition some functions down the line that assume that
            # there's a video that exists will crash. We can send an additional request to the chat in case that happens but
            # that's just decreasing the odds of it happening and not solving the issue. In the mean time I'm just passing the
            # whole video with no cuts

        else:
            all_locs_list = [[(3, 10), (36, 41), (214, 29)],
                            [(0, 58), (59, 29), (152, 52), (207, 41)],
                            [(17, 5), (37, 35), (91, 6), (124, 25), (209, 23)],
                            [(0, 37), (44, 39), (88, 56), (145, 15), (161, 32)],
                            [(5, 10), (16, 13), (62, 18), (101, 20), (132, 23)]]
            all_locs = all_locs_list[ind]
        print(all_locs)
        if len(all_locs) == 0:
            all_locs = [(0, len(df['text']) - 1)]
        time_intervals = [(df['start_times'].iloc[i], df['end_times'].iloc[i + j] + extra_time) for i, j in all_locs]
        # Not elegant and might need reworking but stitches time intervals if one ends and the other starts on the same instance
        # if len(time_intervals) > 1:
        #     final_time_intervals = []
        #     cur_time_interval = time_intervals[0]
        #     for i in range(1, len(time_intervals)):
        #         if time_intervals[i][0] - cur_time_interval[1] <= extra_time:
        #             cur_time_interval = (cur_time_interval[0], time_intervals[i][1])
        #         else:
        #             final_time_intervals.append(cur_time_interval)
        #             cur_time_interval = time_intervals[i]
        #     final_time_intervals.append(cur_time_interval)
        # else:
        #     final_time_intervals = time_intervals

        final_time_intervals = time_intervals
        new_df_inds = []
        ret_df = df.copy()

        for j, x in enumerate(all_locs):

            new_df_inds += list(range(x[0], x[0] + x[1] + 1))

            ret_df['end_times'].iloc[list(range(x[0], x[0] + x[1] + 1))] -= ret_df['start_times'].iloc[x[0]]

            ret_df['start_times'].iloc[list(range(x[0], x[0] + x[1] + 1))] -= ret_df['start_times'].iloc[x[0]]

            if j > 0:
                # prev_end_time = ret_df['end_times'].iloc[all_locs[j - 1][0] + all_locs[j - 1][1] - 1]
                # print(f"Previous end time: {prev_end_time}")
                ret_df['start_times'].iloc[list(range(x[0], x[0] + x[1] + 1))] += ret_df['end_times'].iloc[
                    all_locs[j - 1][0] + all_locs[j - 1][1]]
                ret_df['end_times'].iloc[list(range(x[0], x[0] + x[1] + 1))] += ret_df['end_times'].iloc[
                    all_locs[j - 1][0] + all_locs[j - 1][1]]

        return final_time_intervals, ret_df.iloc[new_df_inds]


    def get_summeraztion_times(self, ind):
        text_part = self.text_parts[ind]
        df = self.dfs[ind]
        assistant_reply = self.call_chat_gpt(ind)
        print(f"Chat GPT reply: {assistant_reply}")
        return self.get_time_intervals(df, assistant_reply, ind)

        # # Debugging time intervals
        # assistant_reply = ""
        # debugging = True
        # return self.get_time_intervals(df, assistant_reply, ind, debugging) 


    def cut_vids(self):
        time_intervals = self.youtube_data.spans
        dfs = self.dfs
        stock_vids = self.stock_videos
        key_words_lst = [sv[1] if sv else None for sv in stock_vids]
        start_times = get_start_times(key_words_lst, dfs)
        filename = self.youtube_data.filename + ".mp4"
        all_vids = []
        for i, x in enumerate(time_intervals):
            new_cut_vids = []
            new_start_time = 0.0
            cut_vid = VideoFileClip(filename).subclip(x[0], x[1])
            for inner_time_interval in self.time_intervals[i]:
                if start_times[i] and inner_time_interval[0] < start_times[i]:
                    if start_times[i] > inner_time_interval[1]:
                        new_start_time += inner_time_interval[1] - inner_time_interval[0]
                    else:
                        new_start_time += start_times[i] - inner_time_interval[0]
                new_cut_vids.append(cut_vid.subclip(inner_time_interval[0], inner_time_interval[1]))
            all_vids.append(concatenate_videoclips(new_cut_vids))
            self.stock_videos[i] = ((self.stock_videos[i][0], self.stock_videos[i][1], new_start_time) if new_start_time != 0.0 else None)
        # # Just do one video at a time
        # original_time = time_intervals[0]
        # cut_vid = VideoFileClip(filename).subclip(original_time[0], original_time[1])
        # new_cut_vids = [cut_vid.subclip(x[0], x[1]) for x in self.time_intervals[0]]
        # all_vids.append(concatenate_videoclips(new_cut_vids))
        return all_vids


    def load_youtube_info(name, dest="downloaded_files/"):
        fn = dest + name + "/object_data.pkl"
        with open(fn, 'rb') as inp:
            obj = pickle.load(inp)
        return obj


if __name__ == "__main__":
    pickled_obj_loc = f"downloaded_files//{SAVED_NAME}//object_data.pkl"
    youtube_obj = pd.read_pickle(pickled_obj_loc)
    EditedVideos(youtube_obj, load_gpt=True)
    
    