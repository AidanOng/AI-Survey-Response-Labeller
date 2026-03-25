# AI-Survey-Response-Labeller

End to end application used to automate labelling of survey responses for my army branch
Classifies 10000+ responses each month at 80% accuracy with a fine-tuned FLAN-TF model.

filter.py - Cleans raw survey responses by removing short text and generic phrases
Filters raw feedback CSVs into:
Relevant responses
Irrelevant responses
Combined dataset with labels

labeller.py - Response classifier
Uses fine-tuned FLAN-T5 model to label survey feedback at ~75% accuracy

plot_data.py - Generates graphs from labelled data
Includes Pie charts, Heatmaps Category tables

app.py - User interface
User will upload CSVs and assign time periods, then app will output filtered and labelled files and charts 
Includes hover preview for for CSVs and graphs
