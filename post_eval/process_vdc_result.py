import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import os
import random
import ast
import argparse
from tqdm import tqdm
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint


camera_caption_prompts = [
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
    "Given these 8 equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
    "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
    "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
    "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
    "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames."
]

detailed_caption_prompts = [
    "The images are given containing 8 equally spaced video frames. Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
    "You are given a sequence of 8 equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
    "The following set contains 8 equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
    "Below are 8 equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
    "A sequence of 8 equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
    "The images provided include 8 equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
    "You are given 8 equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
    "The sequence includes 8 equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
    "The provided images contain 8 equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
    "Here are 8 equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
    "The set of images includes 8 equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
    "Describe the video based on these frames in a few sentences.",
    "What is happening in the video shown in these frames?",
    "Explain the video using these frames.",
    "Imagine the video from these frames and describe it in detail in a few sentences.",
    "Based on these frames, provide a narrative of the video in more than three sentences.",
    "Describe the events in the video shown by these frames in at least three sentences.",
    "Visualize the video from these frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
    "Given these 8 equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
    "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
    "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
    "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences."
    ]

background_caption_prompts = [
    "The images are given containing 8 equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
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
    "Given these 8 equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time of day.",
    "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
    "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
    "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
    "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time."
]

short_caption_prompts = [
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
    "Briefly one-sentence Summary of the visual, Photographic and artistic style."
]

main_object_caption_prompts = [
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
    "Given these 8 equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
    "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
    "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
    "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
    "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames."
]


@function
def gener_pred_response(s, pred_cap, q):
    s += system(
        "You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image."
        "Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Read the detailed description carefully.\n"
        "- Answer the question only based on the detailed description.\n"
        "- The answer should be a short sentence or phrase.\n"
    )
    s += user(
        "Please provide accurate answers to questions related to the content based on a detailed description of a video or image:\n\n"
        f"detailed description: {pred_cap}, question: {q}"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."
    )
    s += assistant(gen("answer_1", max_tokens=256))


@function
def gener_pred_score(s, qa):
    s += system(
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    s += user(
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {qa['question']}\n"
        f"Correct Answer: {qa['answer']}\n"
        f"Predicted Answer: {qa['response']}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )
    s += assistant(gen("answer_1", max_tokens=256))


# --- Main Execution Logic ---
def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Process VDC results and evaluate captions.")
    parser.add_argument('--raw_file', type=str, required=True, help='Path to the raw input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--tp_qa_path', type=str, default='post_eval/background.jsonl', help='Path to the TP QA JSONL file (default: post_eval/background.jsonl).')
    # TODO: Add arguments for captions_path, caption_type, and prompt_list if needed
    args = parser.parse_args()
    # --- End Argument Parser Setup ---


    tp_gt_qa_dict = {}
    # Use the argument for the path
    tp_qa_path = args.tp_qa_path
    with open(tp_qa_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            tp_gt_qa_dict.update(data)

    captions_dict = {}
    captions_path = 'post_eval/VDC_1k_captions.jsonl'
    with open(captions_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            key = data['video_id']
            # Change here
            # TODO: Consider making the caption type ('main_object_caption') an argument
            caption = data['captions']['main_object_caption']
            captions_dict[key] = caption

    set_default_backend(RuntimeEndpoint("http://localhost:30000"))


    # Use arguments instead of hardcoded dictionary
    raw_file = args.raw_file
    output_file = args.output_file


    with open(raw_file, 'r') as gener_file:
        gener_data = json.load(gener_file)


    result_list = []
    tp_scores = []
    tp_accs = []
    pred_dict = {}
    answer_dict = {}


    for idx, meta_data in tqdm(enumerate(gener_data)):
        video_id = meta_data['doc']['video_id']
        # Change here
        pred = meta_data['resps'][0][0]
        answer = captions_dict[video_id]
        pred_dict.update({str(idx):[pred]})
        answer_dict.update({str(idx):[answer]})


        try:
            # TP
            # step 1: generate QA from the ground truth caption
            if video_id in tp_gt_qa_dict.keys():
                result_gtqa_list = tp_gt_qa_dict[video_id]

                tp_result_dict = {
                    'id': video_id,
                    'pred_caption': pred,
                    'gt_caption': answer,
                    'qa_tp_list': []
                }
                # step 2: generate response for each question
                qa_list = []
                for qa_dict in result_gtqa_list:
                    temp_dict = {"question": qa_dict['question'], "answer": qa_dict['answer']}

                    state = gener_pred_response.run(
                        pred_cap=pred,
                        q=qa_dict['question'],
                    )

                    temp_dict["response"] = state["answer_1"]

                    qa_list.append(temp_dict)
                # step 3: match the generated answers with the ground truth answers
                for qa in qa_list:
                    state = gener_pred_score.run(
                        qa=qa,
                    )
                    response_dict = ast.literal_eval(state["answer_1"])

                    qa.update(response_dict)

                # step 4: calculate the final score
                total_score, total_acc = 0, 0
                for qa in qa_list:
                    total_score += float(qa['score'])
                    tp_result_dict['qa_tp_list'].append(qa)
                    if qa['pred'] == 'yes':
                        total_acc += 1
                tp_score = total_score / len(qa_list)
                tp_acc = total_acc / len(qa_list)

                if tp_score and tp_acc:
                    tp_scores.append(float(tp_score))
                    tp_accs.append(float(tp_acc))
                    result_list.append({'tp_score': tp_score, 'tp_acc': tp_acc})
        except Exception as e:
            print(e)

    tp_score = sum(tp_scores) / len(tp_scores)
    tp_acc = sum(tp_accs) / len(tp_accs)


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        for item in result_list:
            file.write(json.dumps(item) + '\n')
        # Append overall scores to the same file
        final_scores = {
            'tp_score': tp_score,
            'tp_acc': tp_acc
        }
        file.write(json.dumps(final_scores) + '\n')

    print(f"Results saved to {output_file}")
    print(f"Overall TP Score: {tp_score}")
    print(f"Overall TP Accuracy: {tp_acc}")


if __name__ == "__main__":
    main()

