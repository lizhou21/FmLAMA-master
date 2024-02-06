### Step 1: Obtain the experimental-result files
Download the experiment results **output** in [Google drive](https://drive.google.com/drive/folders/1lBoYEPafbZHwVYn6u7Em3O5yKM405iRP?usp=sharing)


```
|- output
  |- output_code      # experiments about code-switching analysis (Figure 2)
  |- output_w_lang    # experiments on each sub-dataset FmLAMA-la, filter by each single language. (Table 2, Table 5,6,7,8,9) 
  |- output_wo_filter # experiments on the sub-dataset consisting of 175 food instances universally present across all six involved languages (Figure 3)    
  |- output_wo_lang   # experiments on each sub-dataset FmLAMA-la by using prompts that incorporate cultural information (Figure 1)
  |- output_LLMs      # experiments with the model Bloom, LLaMa, vicuna, chatGPT (Table 3)
```

### Step 2: Read the experimental results

1. `read_resuls.py`: used for read the probing results about BERT-like LLMs and T5-style models. (output_code / output_w_lang / output_wo_filter / output_wo_lang )
   
   e.g., `run read_resuls.py --root_dir output_wo_lang --model mbert_base_cased --lang en --prompt hasParts_1`

2. `read_LLMs_results.py`: used for read the probing results about Bloom, LLaMa, vicuna, chatGPT.

   Bloom prompt 1: Question: What's ingredient in the food [X]? The answer is:
   
   Bloom prompt 2: Question: In [C], what's ingredient in the food [X]? The answer is:

   LLaMa / vicuna / chatGPT prompt 1: The food [X] has the ingredients of []. Please fill the sentence.

   LLaMa / vicuna / chatGPT prompt 2: In [C], the food [X] has the ingredients of []. Please fill the sentence.
   