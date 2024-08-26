import sys

sys.path.append('.')

from langchain_community.chat_models import ChatOllama, ChatLlamaCpp

import argparse
import json
from glob import glob

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

import ipdb
from multiprocessing import Pool
from tqdm import tqdm

from src.langchain_util import init_langchain_model
from src.openie_extraction_instructions import ner_prompts, openie_post_ner_prompts
from src.processing import extract_json_dict

from vllm import LLM, SamplingParams
import pickle

openie_errors = {}

def print_messages(messages):
    for message in messages:
        print(message['content'])


def turn_to_llama3_chat(messages):

    prompt = ''
    
    eot = '<|eot_id|>'
    system = '<|start_header_id|>system<|end_header_id|>'
    user = '<|start_header_id|>user<|end_header_id|>'
    assistant = '<|start_header_id|>assistant<|end_header_id|>'

    for message in messages:
        if message.type == 'system':
            prompt += system + ' ' + message.content + eot + '\n'
        elif message.type == 'human':
            prompt += user + ' ' + message.content + eot + '\n'
        elif message.type == 'ai':
            prompt += assistant + ' ' + message.content + eot + '\n'

    prompt += assistant + ' '
    
    return prompt

def named_entity_recognition(passage: str):
    ner_messages = ner_prompts.format_prompt(user_input=passage)

    total_tokens = 0
    response_content = '{}'

    error = True
    while error:
        try:
            if isinstance(client, ChatOpenAI):  # JSON mode
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0, response_format={"type": "json_object"})
                response_content = chat_completion.content
            elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):
                response_content = client.invoke(ner_messages.to_messages())
            elif isinstance(client, ChatTogether):
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0)
                response_content = chat_completion.content
            else:
                prompt = turn_to_llama3_chat(ner_messages.to_messages())
                sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
                response_content = client.generate(prompt, sampling_params)[0].outputs[0].text

            error = False
        except Exception as e:
            print(e)

    try:
        if isinstance(client, ChatOpenAI):  # JSON mode    
            response_content = eval(response_content)
            total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']
        elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):
            response_content = extract_json_dict(response_content)
            total_tokens += len(response_content.split())
        elif isinstance(client, ChatTogether):
            response_content = extract_json_dict(response_content)
            total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']
        else:
            response_content = extract_json_dict(response_content)
    
        if 'named_entities' not in response_content:
            response_content = []
        else:
            response_content = response_content['named_entities']
    except Exception as e:
        print('Passage NER exception')
        print(e)

    return response_content, total_tokens


def openie_post_ner_extract(passage: str, entities: list, model: str):
    named_entity_json = {"named_entities": entities}
    openie_messages = openie_post_ner_prompts.format_prompt(passage=passage, named_entity_json=json.dumps(named_entity_json))

    response_content = '{}'

    error = True
    while error:
        try:
            if isinstance(client, ChatOpenAI):  # JSON mode
                chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096, response_format={"type": "json_object"})
                response_content = chat_completion.content
                total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
            elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):
                response_content = client.invoke(openie_messages.to_messages())
                total_tokens = len(response_content.split())        
            elif isinstance(client, ChatTogether):
                chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096)
                response_content = chat_completion.content
                total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
            else:                
                prompt = turn_to_llama3_chat(openie_messages.to_messages())
                sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
                response_content = client.generate(prompt, sampling_params)[0].outputs[0].text
                total_tokens = 0

            error = False
        except Exception as e:
            print(e)

    return response_content, total_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='10')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=10)

    args = parser.parse_args()

    dataset = args.dataset
    run_ner = args.run_ner
    num_passages = args.num_passages
    model_name = args.model_name

    corpus = json.load(open(f'data/{dataset}_corpus.json', 'r'))

    if 'hotpotqa' in dataset:
        keys = list(corpus.keys())
        retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(corpus[key])} for i, key in enumerate(keys)]
    else:
        retrieval_corpus = corpus
        for document in retrieval_corpus:
            document['passage'] = document['title'] + '\n' + document['text']

    dataset = '_' + dataset

    if num_passages == 'all':
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except:
            assert False, "Set 'num_passages' to an integer or 'all'"

    flag_names = ['ner']
    flags_present = [flag_names[i] for i, flag in enumerate([run_ner]) if flag]
    if len(flags_present) > 0:
        arg_str = '_'.join(flags_present) + '_' + model_name.replace('/', '_') + f'_{num_passages}'
    else:
        arg_str = model_name.replace('/', '_') + f'_{num_passages}'

    print(arg_str)

    if args.llm == 'vllm':
        client = LLM(model=model_name, tensor_parallel_size=4)
    else:
        client = init_langchain_model(args.llm, model_name)  # LangChain model
        
    already_done = False

    try:
        # Get incomplete extraction output with same settings
        arg_str_regex = arg_str.replace(str(num_passages), '*')

        prev_num_passages = 0
        new_json_temp = None

        for file in glob('output/openie{}_results_{}.json'.format(dataset, arg_str_regex)):
            possible_json = json.load(open(file, 'r'))
            if prev_num_passages < len(possible_json['docs']):
                prev_num_passages = len(possible_json['docs'])
                new_json_temp = possible_json

        existing_json = new_json_temp['docs']
        if 'ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['ents_by_doc']
        elif 'non_dedup_ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['non_dedup_ents_by_doc']
        else:
            ents_by_doc = []

        if num_passages < len(existing_json):
            already_done = True
    except:
        existing_json = []
        ents_by_doc = []

    # Loading files which would reduce API consumption
    aux_file_str = '_'.join(flags_present) + '*_' + model_name + f'_{args.num_passages}'
    aux_file_str = aux_file_str.replace('{}'.format(num_passages), '*')
    auxiliary_files = glob('output/openie{}_results_{}.json'.format(dataset, aux_file_str))

    auxiliary_file_exists = False

    if len(auxiliary_files) > 0:
        for auxiliary_file in auxiliary_files:
            aux_info_json = json.load(open(auxiliary_file, 'r'))
            if len(aux_info_json['docs']) >= num_passages:
                ents_by_doc = aux_info_json["ents_by_doc"]
                auxiliary_file_exists = True
                print('Using Auxiliary File: {}'.format(auxiliary_file))
                break


    def extract_openie_from_triples(triple_json):

        new_json = []
        all_entities = []

        chatgpt_total_tokens = 0

        for i, r in tqdm(triple_json, total=len(triple_json)):

            passage = r['passage']

            if i < len(existing_json):
                triples = existing_json[i]['extracted_triples']
            else:
                triples = None
            
            if triples is not None and len(triples) > 0:
                new_json.append(existing_json[i])
            else:
                if auxiliary_file_exists:
                    doc_entities = ents_by_doc[i]
                else:
                    doc_entities, total_ner_tokens = named_entity_recognition(passage)

                    doc_entities = list(np.unique(doc_entities))
                    chatgpt_total_tokens += total_ner_tokens

                    ents_by_doc.append(doc_entities)

                triples, total_tokens = openie_post_ner_extract(passage, doc_entities, model_name)
                chatgpt_total_tokens += total_tokens

                r['extracted_entities'] = doc_entities

                try:
                    if not(isinstance(client, ChatOpenAI)):
                        triples = extract_json_dict(triples)
                        triples = str(triples)
                    r['extracted_triples'] = eval(triples)["triples"]
                except Exception as e:
                    print('OpenIE exception', e)
                    r['extracted_triples'] = []
                    openie_errors[passage] = triples

                new_json.append(r)

        return (new_json, all_entities, chatgpt_total_tokens)


    extracted_triples_subset = retrieval_corpus[:num_passages]

    num_processes = args.num_processes

    splits = np.array_split(range(len(extracted_triples_subset)), num_processes)

    args = []

    for split in splits:
        args.append([(i, extracted_triples_subset[i]) for i in split])

    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            outputs = pool.map(extract_openie_from_triples, args)
    else:
        outputs = [extract_openie_from_triples(arg) for arg in args]

    new_json = []
    all_entities = []
    lm_total_tokens = 0

    for output in outputs:
        new_json.extend(output[0])
        all_entities.extend(output[1])
        lm_total_tokens += output[2]

    # if not (already_done):
    avg_ent_chars = np.mean([len(e) for e in all_entities])
    avg_ent_words = np.mean([len(e.split()) for e in all_entities])

    # Current Cost
    approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

    extra_info_json = {"docs": new_json,
                       "ents_by_doc": ents_by_doc,
                       "avg_ent_chars": avg_ent_chars,
                       "avg_ent_words": avg_ent_words,
                       "num_tokens": lm_total_tokens,
                       "approx_total_tokens": approx_total_tokens,
                       }
    output_path = 'output/openie{}_results_{}.json'.format(dataset, arg_str)
    json.dump(extra_info_json, open(output_path, 'w'))
    errors_path = 'output/openie{}_errors_{}.p'.format(dataset, arg_str)
    pickle.dump(openie_errors, open(errors_path, 'wb'))
    print('OpenIE saved to', output_path)
