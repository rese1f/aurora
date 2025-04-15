import ast
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import requests
import yaml

import lmms_eval.tasks._task_utils.file_utils as file_utils


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "Test_Videos")

from loguru import logger as eval_logger

DETAILED_CAPTION_PROMPTS = [
    "Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
    "You are given a sequence of equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
    "The following set contains equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
    "Below are equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
    "A sequence of equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
    "The images provided include equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
    "You are given equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
    "The sequence includes equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
    "The provided images contain equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
    "Here are equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
    "The set of images includes equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
    "Describe the video based on these frames in a few sentences.",
    "What is happening in the video shown in these frames?",
    "Explain the video using these frames.",
    "Imagine the video from these frames and describe it in detail in a few sentences.",
    "Based on these frames, provide a narrative of the video in more than three sentences.",
    "Describe the events in the video shown by these frames in at least three sentences.",
    "Visualize the video from these frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
    "Given these equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
    "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
    "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
    "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences.",
]

BACKGROUND_CAPTION_PROMPTS = [
    "The images are given containing equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
    "Describe the background, including objects, location, weather, and time.",
    "Summarize the background setting of the video based on these frames.",
    "What is the environment like in these frames?",
    "Describe the location and weather in these frames.",
    "What background objects and settings are visible in these frames?",
    "Summarize the background of the video, including details about the location, objects, weather, and time.",
    "Describe the environment shown in these frames, covering objects, location, weather, and time.",
    "Provide a detailed background description based on these frames, mentioning objects, location, weather, and time.",
    "Explain the setting of the video, focusing on the background elements like objects, location, weather, and time.",
    "Describe the overall environment in these frames, including details about objects, location, weather, and time.",
    "Given these equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time.",
    "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
    "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
    "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
    "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time.",
]

SHORT_CAPTION_PROMPTS = [
    "Write a one-sentence summary of the video.",
    "Summarize the video in one concise sentence.",
    "Provide a brief description of the video in one sentence.",
    "Describe the main action in the video in one sentence.",
    "What is the video about? Summarize it in one sentence.",
    "In one sentence, summarize the key visual elements of the video.",
    "Provide a one-sentence summary that captures the main subject and action in the video.",
    "Write a concise one-sentence description that encapsulates the essence of the video.",
    "Describe the main theme or action of the video in a single sentence.",
    "What is happening in the video? Provide a one-sentence summary.",
    "Given these frames, write a brief one-sentence summary that captures the essence of the video's visual and artistic style.",
    "Summarize the key visual and thematic elements of the video in one concise sentence.",
    "Provide a one-sentence description that highlights the main subject and action depicted in the video.",
    "In one sentence, describe the primary visual and artistic elements of the video.",
    "Write a concise one-sentence summary that encapsulates the main action and visual style of the video.",
    "Briefly one-sentence Summary of the visual, Photographic and artistic style.",
]

MAIN_OBJECT_CAPTION_PROMPTS = [
    "Description of the main subject actions or status sequence. This suggests including the main subjects (person, object, animal, or none) and their attributes, their action, their position, and movements during the video frames.",
    "Describe the main subject's actions and movements.",
    "What is the main object doing in these frames?",
    "Summarize the primary subject's attributes and actions.",
    "Describe the main subject's position and movements.",
    "What actions does the main object take in these frames?",
    "Describe the main subject, including their attributes and movements throughout the video.",
    "Provide a detailed description of the main object's actions and positions in these frames.",
    "Summarize the main subject's actions, attributes, and movements during the video.",
    "Describe the primary subject's movements and actions in detail.",
    "What are the main object's attributes and how do they move throughout the video?",
    "Given these equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
    "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
    "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
    "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
    "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames.",
]

CAMERA_CAPTION_PROMPTS = [
    "Summary of the view shot, camera movement and changes in shooting angles in the sequence of video frames.",
    "Describe the camera movements in these frames.",
    "What are the camera angles and movements throughout the video?",
    "Summarize the camera actions and perspectives.",
    "Describe any camera zooms, pans, or angle changes.",
    "What camera movements are present in these frames?",
    "Describe the camera's movements, including pans, zooms, and angle changes in these frames.",
    "Summarize the camera actions and changes in shooting angles during the video.",
    "Provide a detailed description of the camera's movements and perspectives.",
    "Describe the camera's actions and how it follows the main subject.",
    "What are the camera movements and angle shifts in these frames?",
    "Given these equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
    "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
    "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
    "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
    "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames.",
]


# Pass in video path here
# Can only work correctly with video llm
def vdc_doc_to_visual(doc):
    video_path = doc["video_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# format the prompt
def vdc_doc_to_text_short(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = random.choice(SHORT_CAPTION_PROMPTS)
    return f"{pre_prompt}"


def vdc_doc_to_text_detailed(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = random.choice(DETAILED_CAPTION_PROMPTS)
    return f"{pre_prompt}"


def vdc_doc_to_text_main_object(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = random.choice(MAIN_OBJECT_CAPTION_PROMPTS)
    return f"{pre_prompt}"


def vdc_doc_to_text_camera(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = random.choice(CAMERA_CAPTION_PROMPTS)
    return f"{pre_prompt}"


def vdc_doc_to_text_background(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = random.choice(BACKGROUND_CAPTION_PROMPTS)
    return f"{pre_prompt}"


def vdc_doc_to_answer(doc):
    return doc["caption"]


# Process result for evaluation in generic task
def vdc_process_results_generic(doc, result):
    pred = result[0]
    doc["pred"] = pred

    return {
        "llm_eval_score": {"video_name": doc["video_name"], "caption": doc["caption"], "pred": pred}
    }


def vdc_pass_through(results, args):
    stored_results = []
    for result in results:
        stored_results.append({"video_name": result["video_id"], "question": result["question"], "answer": result["answer"], "pred": result["pred"]})

    path = generate_submission_file("vdc_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in {path}.")