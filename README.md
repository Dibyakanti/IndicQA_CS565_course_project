# IndicQA_CS565_course_project

This contains the code for the results obtained using mBERT, XLM-Roberta and indic-NLP on chaii dataset.

## Dataset :

The chaii-dataset was directly used for fine-tuning and besides that we used models that have been pre-trained on xQUAD, SQUAD2 and mergedQUAD.\ 
The chai dataset consists of : \
`context` : a paragraph based on which the question has to be answered \
`question` : the question that has to be answered \
`answer_start` : the index from which the answer starts (only train) \
`answer_text` : the answer in string format

## Models :

The models which were used are m-BERT (multilingual BERT), XLM-Roberta and indic-BERT.

## Train :

To train, run the notebooks after setting the proper address for tokenizer and model.

## Test : 



## Results

|       Model       | no-finetuning           | finetuning (with freezing)           | finetuning (without freezing)           |  
| ------------------- | ------------- | ------------- | ------------- |
| mBERT | 0.0373        | 0.1013        | 0.2075        |
| XLM-RoBERTa | 0.2720        | 0.47679        | 0.5837        |
| indic-BERT | 0.0016        | -        | 0.0288        |
