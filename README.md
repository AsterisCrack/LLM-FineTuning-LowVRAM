# LLM-FineTuning-LowVRAM
A project aimed to fine-tune a pretrained LLM with single GPU and low VRAM environments

## Tutorial
First, install the required libraries by running:
```
pip install -r requirements.txt
```
To train the model, go to src/train.py and modify any hyperparameters you want. Then, you can start training by running:
```
python src/train.py
```
You can also use src/train_no_deepspeed.py if you have at least 16 GB of GPU memory. \
Also, you may run src/optuna_parameter_search.py if you have time to search for the best hyperparameters. \
\
If you stopped training and want to keep training from a saved checkpoint go to src/train_resume_training.py and change the path to the checkpoint. Then run the following command to keep training from that point:
```
python src/train_resume_training.py
```

## Use model
Once trained, go to src/use_model.py and change the location of the trained model.
You can then execute the following command to use the model:
```
python src/use_model.py
```
We encourage playing with the sampling parameters to look for the best combination for your use case. \
Our trained model can be loaded from huggingface from:
```
Asteris/qwen2.5-7B-LoRa-efficient-training
```

## IFEval Results
To execute IFEval first run the notebook provided. Once you have the results run:
```
python -m instruction_following_eval.evaluation_main --input_data instruction_following_eval/data/input_data.jsonl --input_response_data instruction_following_eval/data/input_response_data.jsonl --output_dir instruction_following_eval/data/evaluation_results
```
Our results were: \
Strict Accuracy Scores: \
prompt-level: 0.3111111111111111 \
instruction-level: 0.4495192307692308 \
 \
change_case 0.20224719101123595 \
combination 0.6153846153846154 \
detectable_content 0.9615384615384616 \
detectable_format 0.6153846153846154 \
keywords 0.50920245398773 \
language 0.9032258064516129 \
length_constraints 0.32867132867132864 \
punctuation 0.12121212121212122 \
startend 0.05970149253731343 \
 \
change_case:capital_word_frequency 0.68 \
change_case:english_capital 0.0 \
change_case:english_lowercase 0.02564102564102564 \
combination:repeat_prompt 0.8048780487804879 \
combination:two_responses 0.2916666666666667 \
detectable_content:number_placeholders 0.9230769230769231 \
detectable_content:postscript 1.0 \
detectable_format:constrained_response 1.0 \
detectable_format:json_format 0.0 \
detectable_format:multiple_sections 0.9285714285714286 \
detectable_format:number_bullet_lists 0.0 \
detectable_format:number_highlighted_sections 0.7659574468085106 \
detectable_format:title 1.0 \
keywords:existence 1.0 \
keywords:forbidden_words 0.0 \
keywords:frequency 0.5952380952380952 \
keywords:letter_frequency 0.5757575757575758 \
language:response_language 0.9032258064516129 \
length_constraints:nth_paragraph_first_word 0.16666666666666666 \
length_constraints:number_paragraphs 0.0 \
length_constraints:number_sentences 0.38461538461538464 \
length_constraints:number_words 0.4807692307692308 \
punctuation:no_comma 0.12121212121212122 \
startend:end_checker 0.15384615384615385 \
startend:quotation 0.0 \
 \
================================================================ \
Loose Accuracy Scores: \
prompt-level: 0.3925925925925926 \
instruction-level: 0.53125 \
 \
change_case 0.39325842696629215 \
combination 0.7692307692307693 \
detectable_content 0.9615384615384616 \
detectable_format 0.6153846153846154 \
keywords 0.6319018404907976 \
language 0.9032258064516129 \
length_constraints 0.40559440559440557 \
punctuation 0.22727272727272727 \
startend 0.1044776119402985 \
 \
change_case:capital_word_frequency 0.72 \
change_case:english_capital 0.2 \
change_case:english_lowercase 0.3076923076923077 \
combination:repeat_prompt 0.975609756097561 \
combination:two_responses 0.4166666666666667 \
detectable_content:number_placeholders 0.9230769230769231 \
detectable_content:postscript 1.0 \
detectable_format:constrained_response 1.0 \
detectable_format:json_format 0.0 \
detectable_format:multiple_sections 0.9285714285714286 \
detectable_format:number_bullet_lists 0.0 \
detectable_format:number_highlighted_sections 0.7659574468085106 \
detectable_format:title 1.0 \
keywords:existence 1.0 \
keywords:forbidden_words 0.2857142857142857 \
keywords:frequency 0.6904761904761905 \
keywords:letter_frequency 0.6363636363636364 \
language:response_language 0.9032258064516129 \
length_constraints:nth_paragraph_first_word 0.16666666666666666 \
length_constraints:number_paragraphs 0.25925925925925924 \
length_constraints:number_sentences 0.40384615384615385 \
length_constraints:number_words 0.5384615384615384 \
punctuation:no_comma 0.22727272727272727 \
startend:end_checker 0.15384615384615385 \
startend:quotation 0.07317073170731707