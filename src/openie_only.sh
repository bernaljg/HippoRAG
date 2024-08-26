data=$1  # e.g., 'sample'
extraction_model=$2 # e.g., 'gpt-3.5-turbo-1106' (OpenAI), 'meta-llama/Llama-3-8b-chat-hf' (Together AI)
available_gpus=$3
llm_api=$4 # e.g., 'openai', 'together'
extraction_type=ner

# Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model --run_ner --num_passages 10 --num_processes 10 # NER and OpenIE for passages
