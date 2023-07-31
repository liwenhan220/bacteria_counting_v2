# bacteria_counting_v2

This is an improved (I think) version of my previous bacteria counting project

# Results (Preview)

# Prerequisites

Have Pytorch, Matplotlib, numpy installed. The following are steps to train your bacteria model

# Step 1: extract bacteria from raw images and store them as bacteria

run `python generate_data.py`, the resulting bacteria will be stored in "extracted_data" folder. To view which bacteria are extracted, check images in "debug" folder.

# Step 2: train a model

run `python train_model.py -e [train_epochs] -m [output_model_dir] -l [output_log_dir]`. It will gather the bacteria extracted in previous step and train a simple CNN on them to distinguish between bacteria and residues.

# Step 3: apply your model

run `python label_img.py -m [input_model_dir] -i [input_image_path] -o [output_image_path] -n [input_model_number]`, it will draw bacteria on your input image. Residues will be colored blue, bacteria will be colored red.
