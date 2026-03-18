#region IMPORTS

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

#endregion

#region SETTINGS

#MODELS = ["C:\Users\user\OneDrive\Documents\Programming\Python\AI survey processor\trainer\finetuned-flan-t5"]
#MODEL_PATH = MODELS[0]

CATEGORISED_FILES = ["categorised_data.csv", "labelled pict june.csv", "labelled psict jan - may.csv", "hlabi 25 aug.csv", "lab_nov_2025.csv"]
CATEGORISED_INPUTS = [CATEGORISED_FILES[4]] 
CATEGORISED_DATES = ["November"]
CATEGORISED_INPUTS = ["Category Output//" + i for i in CATEGORISED_INPUTS]

PLOT_INPUT = CATEGORISED_INPUTS # CSV: No column header, 1st column: Responses, 2nd column: Categories
PLOT_PIE = True
PIE_OUTPUT = "Plot Output//responses_pie" # PNG: Pie Chart, each slice is number of responses per category

PLOT_HEATMAP = False
HEATMAP_OUTPUT = "Plot Output//responses_heatmap.png" #PNG: Heatmap, each square is number of responses per category per month

TABULATE_DATA = False
TABLE_OUTPUT = "Plot Output//table_responses.csv" # CSV: 1st column: Categories, 2nd Column: Responses per Category

ALL_CATEGORIES = [
    "In-Pro/Out-Pro",
    "ICT History/ORNS",
    "Book In/Book Out",
    "Parade State",
    "Survey",
    "Training Program",
    "Feature Request"
]

for i in range(0, len(ALL_CATEGORIES)):
    ALL_CATEGORIES[i] = ALL_CATEGORIES[i].lower()
print(ALL_CATEGORIES)


    
        
#region PLOT
def plot_data(input_csvs):
    headers = ["Time Period", "Diagram"]
    info_table = []

    if PLOT_PIE == True:
        pie_files = plot_pies(input_csvs)
        for file_name in pie_files:
            info_table.append(file_name)

    if PLOT_HEATMAP == True or TABULATE_DATA == True:
        table_df = make_table(input_csvs)

    if PLOT_HEATMAP == True:
        heatmap_file = plot_heatmap(table_df)
        info_table.append(["All", heatmap_file])

    if TABULATE_DATA == True:
        table_file = tabulate_data(table_df)
        info_table.append(["All", table_file])

    print(tabulate(info_table, headers, tablefmt="grid"))

def plot_pies(input_csvs):
    file_locations = []

    for csv in input_csvs:
        file_location = plot_pie(csv)
        file_locations.append(file_location)

    return file_locations

def plot_pie(labelled_csv, output_name):
    cat_count = interleave_count(count_responses(labelled_csv))

    responses = np.array(cat_count[0], dtype=int)   
    categories = np.array(cat_count[1], dtype=str)
    filtered_categories = [cat for cat, val in zip(categories, responses) if val > 0]
    filtered_responses = [val for val in responses if val > 0]
    total = np.sum(responses)

    fig, ax = plt.subplots()

    wedges, _ = ax.pie(
        filtered_responses,
        labels=None,   # don’t let pie place labels
        startangle=90,
        wedgeprops=dict(width=0.45),
        textprops={'fontsize': 8}
    )

    # place category + number stacked outside each wedge
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        percentage = round(filtered_responses[i] / total * 100, 2)

        # category label
        ax.text(1.4 * x, 1.4 * y, filtered_categories[i].title(),
                ha='center', va='bottom', fontsize=9)

        # number just below it
        ax.text(1.4 * x, 1.4 * y - 0.03,
                str(filtered_responses[i]),
                ha='center', va='top', fontsize=9, color='black')
        
        ax.text(1.4 * x, 1.4 * y - 0.13,
                str(percentage) + "%",
                ha='center', va='top', fontsize=9, color='black')


    #time_period = CATEGORISED_DATES[CATEGORISED_INPUTS.index(labelled_csv)]
    #OUTPUT = PIE_OUTPUT + "_" + time_period + ".png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    #return [time_period, OUTPUT[12:]]

def absolute_count(pct, all_vals):
        total = sum(all_vals)
        val = int(round(pct*total/100.0))
        return f"{val}"

def plot_heatmap(table_df):
    data = table_df.set_index("Category")

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="Reds", aspect="auto")

    plt.xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
    plt.yticks(range(len(data.index)), data.index)

    plt.colorbar(label="Number of Responses")

    for (i, j), val in np.ndenumerate(data.values):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(HEATMAP_OUTPUT)
    plt.close()

    return HEATMAP_OUTPUT
    

def interleave_count(response_dict) -> tuple: #inserts smaller values between larger values to prevent label overlap 
    categories = list(response_dict.keys())
    responses = list(response_dict.values())

    sorted_pairs = sorted(zip(responses, categories), reverse=True)
    responses_sorted, categories_sorted = zip(*sorted_pairs)

    left = 0
    right = len(responses_sorted) - 1
    interleaved = []

    while left <= right:
        interleaved.append((responses_sorted[left], categories_sorted[left]))

        if left != right:
            interleaved.append((responses_sorted[right], categories_sorted[right]))

        left += 1
        right -= 1

    return list(zip(*interleaved))

def count_responses(labelled_csv) -> tuple:
    labelled_df = pd.read_csv(labelled_csv, names=["response", "category"])

    cat_count = {category : 0 for category in ALL_CATEGORIES}

    for category in labelled_df["category"]:
        if category in ALL_CATEGORIES:
            cat_count[category] += 1

    print(cat_count)
    return cat_count

def make_table(labelled_batch):
    all_counts = {}

    for date, file in labelled_batch.items():
        cat_count = count_responses(file)
        all_counts[date] = cat_count

    table_df = pd.DataFrame(all_counts).fillna(0).astype(int)
    table_df = table_df.reset_index().rename(columns={"index": "Category"})
    
    return table_df

def tabulate_data(table_df):    
    table_df.to_csv(TABLE_OUTPUT, index=False)
    return TABLE_OUTPUT

def plot_batch(labelled_batch):
    plotted_batch_info = {"pie": {}, "heatmap": None, "table": None}

    for date, file in labelled_batch.items():
        output_file = "Plot Output//pie_" + str(date) +".png"
        plot_pie(file, output_file)
        plotted_batch_info["pie"][date] = output_file

    table_df = make_table(labelled_batch)
    plotted_batch_info["heatmap"] = plot_heatmap(table_df)
    plotted_batch_info["table"] = tabulate_data(table_df)

    return plotted_batch_info

#endregion

def main():
    print(plot_pie(CATEGORISED_INPUTS[0], "pie__25.png"))

if __name__ == "__main__":
    main()