#region IMPORTS

import pandas as pd
import re
from time import perf_counter
from tabulate import tabulate

#endregion

#region SETTINGS

UNLABELLED_FILES = ["Feb Unfiltered 26.csv", "unfiltered psict 2024 500.csv", "unf jan - may 25.csv", "unf aug.csv", "Raw September.csv", "Raw October 25.csv"]
UNLABELLED_INPUTS = UNLABELLED_FILES # CSV: No column header, 1st column: Responses
UNLABELLED_DATES = ["nov", "2024", "may", "aug"]
UNLABELLED_INPUTS = ["Unlabelled\\" + i for i in UNLABELLED_INPUTS]
UNLABELLED_BATCH = dict(zip(UNLABELLED_DATES, UNLABELLED_INPUTS))

FILTER_DATA = True
FILTER_INPUT = UNLABELLED_INPUTS # CSV: No column header, 1st column: Responses
FILTER_OUTPUT = "Filter Output\\fil" # CSV: No column header, 1st column: Responses (Relevant only)
IRRELEVANT_OUTPUT = "Filter Output\\irr" # CSV: No column header, 1st column: Responses (Irrelevant only)
COMBINED_OUTPUT = "Filter Output\\com" # CSV: No column header, 1st column: Responses (Relevant and Irrelevant), 2nd column: Categories

#endregion

#region FILTER
def format_batch_for_filtering(entries):
    dates = list(entries.keys())
    formatted_entries = {}

    for date in dates:
        formatted_entries[date] = "Unlabelled\\" + entries[date]

    return formatted_entries

def filter_batch(response_batch: dict):
    dates = list(response_batch.keys())
    filtered_batch_info = {}

    for date in dates:
        filtered_csv_info = filter_csv(response_csv=response_batch[date], date=date)
        filtered_batch_info[date] = filtered_csv_info

    return filtered_batch_info

def print_filtered_batch_info(batch_info):
    dates = ["Date"] + list(batch_info.keys())

    filtered_row = ["Filtered"]
    irrelevant_row = ["Irrelevant"]
    combined_row = ["Combined"]

    for date in dates[1:]:
        filtered_row.append(batch_info[date]["filtered"][0] + f" ({batch_info[date]["filtered"][1]})")
        irrelevant_row.append(batch_info[date]["irrelevant"][0] + f" ({batch_info[date]["irrelevant"][1]})")
        combined_row.append(batch_info[date]["combined"][0] + f" ({batch_info[date]["combined"][1]})")
        

    info_table = [dates, filtered_row, irrelevant_row, combined_row]

    print(tabulate(info_table, dates, tablefmt="grid"))

    return info_table

def filter_csv(response_csv, date) -> dict:
    raw_df = pd.read_csv(response_csv, header=None, names=["response"])

    filtered = []
    irrelevant = []
    combined = []

    for response in raw_df["response"]:
        if is_irrelevant(response):
            irrelevant.append(response)
            combined.append({"response": response, "label": "Irrelevant"})
        
        if not is_irrelevant(response):
            filtered.append(response)
            combined.append({"response": response, "label": "Relevant"})

    FILTERED_CSV = FILTER_OUTPUT + "_" + date + ".csv"
    IRRELEVANT_CSV = IRRELEVANT_OUTPUT + "_" + date + ".csv"
    COMBINED_CSV = COMBINED_OUTPUT + "_" + date + ".csv"

    pd.DataFrame(filtered).to_csv(FILTERED_CSV, index=False, header=False)
    pd.DataFrame(irrelevant).to_csv(IRRELEVANT_CSV, index=False, header=False)
    pd.DataFrame(combined).to_csv(COMBINED_CSV, index=False, header=False)

    filtered_csv_info = {"filtered": (FILTERED_CSV, len(filtered)),
                         "irrelevant": (IRRELEVANT_CSV, len(irrelevant)),
                         "combined": (COMBINED_CSV, len(combined))}

    return filtered_csv_info

#region FILTER DETAILS

# "nil", "nill", "na", "n a", "n.a", "n.a.", "n. a.", "no", "nope", "nah", "none", "nothing", "non", "nik", "ni", "naaa", "nop",

IRRELEVANT_RESPONSES = {
    "none so far", "nothing at all",
    "no comment", "no comments",
    "not at the moment", "not really", "not applicable",
    "no feedback"
}

def is_irrelevant(response) -> bool:
    if not isinstance(response, str):
        return True
    
    cleaned = normalise_text(response)

    if len(cleaned) <= 8:
        return True
    
    elif cleaned in IRRELEVANT_RESPONSES:
        return True
    
    return False

def normalise_text(text: str) -> str:
    text = text.strip()
    text= text.lower()
    text = re.sub(r"[.,!?]+$", "", text)
    
    return text

#endregion

#endregion

def main():
    print(filter_csv(UNLABELLED_INPUTS[0], "Feb 26"))

if __name__ == "__main__":
    main()