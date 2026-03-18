import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

current_run = "_run3"
output_folder_type = "flan_feedback_cls"
output_folder_name = output_folder_type + current_run

fresh_feedback = "hlabi 25 aug.csv"
output_categories = "alabi 25 aug.csv"

LABELS = [
    "training program","parade state","feature request",
    "book in/book out","in-pro/out-pro","survey","ict history/orns", "irrelevant"
]

MODEL_DIR = output_folder_name + "/best"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def label_batch(response_batch: dict):
    labelled_batch_info = {}

    for date in response_batch:
        labelled_output = "Category Output\\lab_" + str(date) + ".csv"
        labelled_batch_info[date] = labelled_output

        response_csv = response_batch[date]
        new_df = pd.read_csv(response_csv, header=None, names=["Response"])
        new_df["Category"] = new_df["Response"].apply(classify)
        new_df.to_csv(labelled_output, index=False)

    return labelled_batch_info

def make_prompt(text):
    return (f"Classify the feedback into one of: {', '.join(LABELS)}.\n\n"
            f'Feedback: "{text}"\n\n'
            f"Answer with only the label.")

def classify(text):
    prompt = make_prompt(text)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8)
    pred = tok.decode(out[0], skip_special_tokens=True).strip().lower()
    return pred

#print(label_batch({"august_25": "Filter Output\\fil_august_25.csv"}))


#print(f"Starting classification of {len(new_df)} categories...")
#print("Done!")

#print(f"Saved predictions to {output_categories}")

#new_df["categories"] = new_df["categories"].str.lower()
#correct = 0
#matches = new_df["categories"] == new_df["predicted_category"]
#counter = matches.sum()

#print (f"Accuracy: {counter/len(new_df)}% ({counter}/{len(new_df)})")
