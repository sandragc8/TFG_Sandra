import pickle, requests, re, gc, os, time, torch, json
import requests, traceback
import pandas as pd
import typing as tp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging
from transformers.utils import hub
from huggingface_hub import file_download as hface
from torch.nn.functional import pad
from statistics import mean
from math import ceil
from json import JSONDecoder
from tqdm import tqdm



class AutoAskerException(Exception):
    pass


class AutoAskerTranslator:

    def __init__(self, excelfile: str, model_config: dict, prompts: list, languages: list):
        self.excel = excelfile
        self.batch_size = model_config.pop('batch_size', 1)
        model_id = model_config.pop('model_id')
        self.model_name = model_id.split(sep='/')[1]
        self.prompt_paraphrase = prompts[0]
        self.prompt_evaluate = prompts[1]
        self.languages = languages
        access_token = "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"
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token, padding_side='left', cache_dir="/home/jovyan/cached_models")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for bits, args in self.get_possible_quantizations(**model_config).items():
            try:
                if args:
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=access_token, cache_dir="/home/jovyan/cached_models", **args)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=access_token, cache_dir="/home/jovyan/cached_models")
                if str(bits) == 'default' or bits > 8:
                    self.model.to('cuda')
                self.model_bits = bits
                break
            except Exception as e:
                traceback.print_exc()
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                if isinstance(e, KeyboardInterrupt):
                    raise KeyboardInterrupt
                else:
                    time.sleep(10)
    
        if not self.model:
            print("AutoAskerException")
            raise AutoAskerException()

    def get_possible_quantizations(self,  b_params: float, max_memory_gb: int = 40, default_config: bool = True):
        d = {32:{'torch_dtype':torch.float32}, 16:{'torch_dtype':torch.float16}, 8:{'load_in_8bit':True}, 4:{'load_in_4bit':True}}
        for bits in d.copy():
            if (float(bits*b_params)/10 > max_memory_gb):
                d.pop(bits, None)
        if default_config:
            d = {**{'default': {}}, **d}
        return d

    def filter_json(self, text, decoder=JSONDecoder()):
        pos = 0
        result = None
        while True:
            match = text.find('{', pos)
            if match == -1:
                return result
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                #yield result
                pos = match + index
            except ValueError:
                pos = match + 1

    def find_json(self, text):
        """
        Busca un objeto JSON dentro de un texto dado y lo devuelve como un diccionario de Python.
        
        Parameters:
        text (str): El texto en el cual buscar el objeto JSON.
    
        Returns:
        dict: El objeto JSON encontrado como un diccionario de Python o None si no se encontró un JSON válido.
        """
        # Expresión regular para encontrar posibles JSON (patrón bastante relajado)
        json_regex = re.compile(r'\{[^{}]*\}')
        
        def try_parse_json(s):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None
            
        def correct_json_string(s):
            # Agrega comillas a las claves y ajusta valores entre comillas si es necesario
            s = re.sub(r'(?<={|,)\s*(\w+)\s*:', r'"\1":', s)
            s = re.sub(r':\s*([^",}\s]+)\s*', r': "\1"', s)
            return s
            
        matches = json_regex.findall(text)
        for match in matches:
            # Intentar directamente parsear el JSON
            parsed_json = try_parse_json(match)
            if parsed_json is not None:
                return parsed_json
            
            # Intentar corregir el JSON y volver a parsearlo
            corrected_json_str = correct_json_string(match)
            parsed_json = try_parse_json(corrected_json_str)
            if parsed_json is not None:
                return parsed_json
        return None
    
    def ask_model_batches(self, questions: list, prompts: list):
        concatenated_tensors = []
        tensor_shapes = []
        total_questions = questions.copy()
        res = {}
        terminators = [
                self.tokenizer.eos_token_id,
        ]
        
        if self.model_name in ('Yi-6B-Chat', 'Yi-34B-Chat'):
            #self.tokenizer.padding_side = 'right'
            for question in total_questions:
                tensor = self.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
                concatenated_tensors.append(tensor)
                tensor_shapes.append(tensor.shape[1])
            num_padding = max(tensor_shapes)
            concatenated_tensors = [pad(input=i, pad=(num_padding-i.shape[1], 0), mode='constant', value=0) for i in concatenated_tensors]
            
        elif self.model_name not in ('bloomz-7b1', 'bloomz-7b1-mt', 'bloomz-7b1-mt-sft-chat', 'gemma-7b-it', 'bertin-gpt-j-6B', 'bertin-gpt-j-6B-alpaca'):
            for question in total_questions:
                tensor = self.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
                concatenated_tensors.append(tensor)
            total_questions = []
            for i in range(len(concatenated_tensors)):
                total_questions.append(self.tokenizer.batch_decode(concatenated_tensors[i])[0])

        if self.model_name in ('Meta-Llama-3-8B-Instruct'):
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        start_time = time.time()
        if self.model_name in ('Yi-6B-Chat', 'Yi-34B-Chat'):
            start_time = time.time()
            outputs = self.model.generate(torch.cat(concatenated_tensors, axis=0), max_new_tokens=2000, do_sample=False)
            lst = self.tokenizer.batch_decode(outputs[:, num_padding:], skip_special_tokens=True)
        else:
            start_time = time.time()
            inputs = self.tokenizer(total_questions, return_tensors="pt", padding=True).to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=2000, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=terminators, do_sample=False)
            lst = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        end_time = time.time()

        token_lst = [i.tolist() for i in inputs['input_ids']] if self.model_name not in ('Yi-6B-Chat', 'Yi-34B-Chat') else [i.tolist()[0] for i in concatenated_tensors]
        res.update({'tokens_in': token_lst})
        res.update({'tokens_in_count': [len(i) for i in token_lst]})
        #lst = [self.purge_response(i) for i in lst]
        token_lst = [self.tokenizer.encode(i) for i in lst]
        res.update({'time': end_time-start_time})
        res.update({'tokens_out': token_lst})
        res.update({'tokens_out_count': [len(i) for i in token_lst]})
        res.update({'word_count': [len(i.split()) for i in lst]})
        res.update({'responses':[lst[i:i+len(prompts)] for i in range(0, len(lst), len(prompts))]})
        return res

    def loop(self, excel_writer, retranslate):
        start_time = end_time = 0
        batch_sizes = range(1,9,1)
        #batch_sizes = [1]
        pbar_lang = tqdm(range(len(self.languages)+1), desc=f'{self.model_name} langs: ', colour='green')
        pbar_lang.n = 0
        pbar_lang.refresh()
        for lang,prompts in self.prompts_langs.items():
            df_time = pd.DataFrame(columns=['Batch Size', 'Experiment Time (s)', 'Prompt Time (s/prompt)', 'Inference Speed (prompt/s)'])
            try:
                df = pd.read_excel(self.excel, sheet_name=lang, engine='openpyxl')
                df_lang = pd.DataFrame(columns=['question', 'option_a', 'option_b', 'option_c', 'option_d', 
                                                'correct_answer', 'llm_question', 'tokens_in', 'tokens_in_count', 'llm_answer', 'tokens_out', 'tokens_out_count', 'word_count',
                                                'llm_answer_filtered'])
                pbar_df = tqdm(range(len(df)), desc=f'Batch_size ({batch_size}-{len(batch_sizes)}): ', colour='green')
                pbar_df.n = 0
                pbar_df.refresh()
                i = 0
                last_progress = 0
                sum_time = 0
                while i < len(df):
                    instructions = []
                    b = 0
                    while i < len(df) and b < batch_size:
                        for prompt in prompts:
                            complete_question = prompt.replace('<question>', df.loc[i]['instruction']) #cambiar "question" a "instruction"
                            for option in ['option_a', 'option_b', 'option_c', 'option_d']:
                                complete_question = complete_question.replace(f'<{option}>', df.loc[i][option])
                            instructions.append(complete_question)
                        i += 1
                        b += 1
                    res_dict = self.ask_model_batches(instructions, prompts)
                    sum_time += res_dict['time']
                    for index, r in enumerate(res_dict['responses']):
                        orig_row = df.loc[i-batch_size+index].tolist()[:-1] # cambiar a :-1
                        df_lang.loc[len(df_lang)] = orig_row + [instructions[index]] + [res_dict['tokens_in'][index],
                                                                    res_dict['tokens_in_count'][index]] + r + [res_dict['tokens_out'][index], 
                                                                                                            res_dict['tokens_out_count'][index], 
                                                                                                            res_dict['word_count'][index]]
                    pbar_df.n = i
                    pbar_df.refresh()
                    #last_progress = self.print_percent(i, len(df), last_progress, f"{self.model_name} lang_{lang} batch_({batch_size})")
                prompt_time = sum_time/len(df)
                df_time.loc[len(df_time)] = [batch_size, sum_time, prompt_time, 1/prompt_time]
                df_time.to_excel(excel_writer, sheet_name=f'{lang}_time', index=False)
                df_lang.to_excel(excel_writer, sheet_name=f'{lang}_resp_{batch_size}', index=False)
            except (Exception, KeyboardInterrupt) as e:
                df_time.to_excel(excel_writer, sheet_name=f'{lang}_time', index=False)
                df_lang.to_excel(excel_writer, sheet_name=f'{lang}_resp_{batch_size}', index=False)
                #print(df_time)
                #print(df_lang)
                traceback.print_exc()
            #df_time.to_excel(excel_writer, sheet_name=f'{lang}_time', index=False)
            pbar_lang.n += 1
            pbar_lang.refresh()

    def paraphrase(self, dataset, language: tuple):
        pbar_iter = tqdm(range(len(dataset)), desc=f'{language[0]}_paraphrase: ', colour='green')
        pbar_iter.n = 0
        pbar_iter.refresh()
        sum_time = 0
        i = 0
        dcolumns = ['correct_answer']
        if 'raw_paraphrase' in dataset.columns:
            dcolumns.append('raw_paraphrase')
        icolumns = dataset.columns[:-1] if 'raw_paraphrase' in dataset.columns else dataset.columns[:-1].insert(0, 'raw_paraphrase')
        mmlu = dataset.copy().drop(columns=dcolumns).to_dict(orient='records')
        df = pd.DataFrame(columns=icolumns)
        while i < len(dataset):
            instructions = []
            b = 0
            while i < len(dataset) and b < self.batch_size:
                mmlu_question = mmlu.pop(0)
                complete_question = self.prompt_paraphrase.replace('<json>', json.dumps(mmlu_question))
                instructions.append(complete_question)
                i += 1
                b += 1
            res_dict = self.ask_model_batches(instructions, [self.prompt_paraphrase])
            sum_time += res_dict['time']
            for index, r in enumerate(res_dict['responses']):
                llm_answer = self.find_json(r[0])
                '''
                if llm_answer is None or (any(value is None for value in llm_answer.values())):
                    print(f'\n{instructions[index]}')
                    print(f'\n{r[0]}')
                    print(f'\n{llm_answer}')
                '''
                if llm_answer:
                    llm_answer.update({'raw_paraphrase': r[0]})
                else:
                    llm_answer = [r[0]] + [None for i in range(len(df.columns)-1)]
                df.loc[len(df)] = llm_answer
            pbar_iter.n = i
            pbar_iter.refresh()
        prompt_time = sum_time/len(df)
        df['correct_answer'] = dataset['correct_answer']
        return df, [sum_time, prompt_time, 1/prompt_time]

    def filter_answer(self, llm_answer, options):
        res = None if llm_answer is None else llm_answer.get('answer', None)
        key_list = list(options.keys())
        val_list = list(options.values())
        if res and res not in key_list:
            try:
                res = key_list[val_list.index(res)]
            except:
                #print(f'llm_answer not in options... {llm_answer}')
                res = None
        return res

    def evaluate(self, dataset, language: tuple):
        pbar_iter = tqdm(range(len(dataset)), desc=f'{language[0]}_evaluation: ', colour='green')
        pbar_iter.n = 0
        pbar_iter.refresh()
        sum_time = 0
        i = 0
        dcolumns = ['correct_answer']
        if 'raw_paraphrase' in dataset.columns:
            dcolumns.append('raw_paraphrase')
        mmlu = dataset.copy().drop(columns=dcolumns).to_dict(orient='records')
        df = pd.DataFrame(columns=[*dataset.columns, 'llm_question', 'tokens_in', 'tokens_in_count', 'llm_answer', 'tokens_out', 'tokens_out_count', 'word_count',
                                                'llm_answer_filtered'])
        while i < len(dataset):
            instructions = []
            b = 0
            while i < len(dataset) and b < self.batch_size:
                mmlu_question = mmlu.pop(0)
                complete_question = self.prompt_evaluate.replace('<json>', json.dumps(mmlu_question))
                instructions.append(complete_question)
                i += 1
                b += 1
            res_dict = self.ask_model_batches(instructions, [self.prompt_paraphrase])
            sum_time += res_dict['time']
            for index, r in enumerate(res_dict['responses']):
                orig_row = dataset.loc[i-b+index].tolist()
                optionsk = dataset.columns[1:-1]
                optionsv = orig_row[1:-1]
                llm_answer = self.find_json(r[0])
                '''
                if llm_answer is None:
                    print(f'\n{instructions[index]}')
                    print(f'\n{r[0]}')
                    print(f'\n{llm_answer}')
                '''
                df.loc[len(df)] = orig_row + [instructions[index]] + [res_dict['tokens_in'][index],
                                                            res_dict['tokens_in_count'][index], 
                                                            json.dumps(llm_answer), res_dict['tokens_out'][index], 
                                                            res_dict['tokens_out_count'][index], res_dict['word_count'][index],
                                                            self.filter_answer(llm_answer, {optionsk[i]: optionsv[i] for i in range(len(optionsk))})]
            
            pbar_iter.n = i
            pbar_iter.refresh()
        prompt_time = sum_time/len(df)
        correct_answers = (df['correct_answer'] == df['llm_answer_filtered']).sum()
        null_answers = df['llm_answer_filtered'].isna().sum()
        wrong_answers = ((df['correct_answer'] != df['llm_answer_filtered']) & (~df['llm_answer_filtered'].isna())).sum()
        accuracy = (float(correct_answers)/len(df))*100
        return df, [sum_time, prompt_time, 1/prompt_time], [correct_answers, accuracy, wrong_answers, null_answers]


    def run(self, result_folder: str = '.', retranslate: bool = True, mid_evaluation: bool = True, lines: int = 0, extra_name: str = ""):
        with pd.ExcelWriter(f'{result_folder}/paraphrases_{self.model_name}_{str(self.model_bits)}bits{extra_name}.xlsx') as excel_writer:
            df_time = pd.DataFrame(columns=['Experiment', 'Experiment Time (s)', 'Prompt Time (s/prompt)', 'Inference Speed (prompt/s)'])
            df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
            df_eval = pd.DataFrame(columns=['Experiment', 'Number of correct answers', 'Accuracy (%)', 'Number of wrong answers', 'Number of null answers'])
            df_eval.to_excel(excel_writer, sheet_name=f'eval_stats', index=False)
            try:
                pbar_lang = tqdm(range(len(self.languages)), desc=f'{self.model_name} langs: ', colour='green')
                pbar_lang.n = 0
                pbar_lang.refresh()
                orig_language = self.languages.pop(0)
                mmlu = pd.read_excel(self.excel, sheet_name=orig_language[0], engine='openpyxl')
                mmlu = mmlu.drop(columns=['id']).rename(columns={"instruction": "question", "option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D", "answer": "correct_answer"})
                if lines != 0:
                    mmlu = mmlu.head(min(len(mmlu), lines))
                eval, time_stats, eval_stats = self.evaluate(mmlu, orig_language)
                eval.to_excel(excel_writer, sheet_name=f'{orig_language[0]}_l0', index=False)
                df_time.loc[len(df_time)] = [f'{orig_language[0]}_l0_eval'] + time_stats
                df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
                df_eval.loc[len(df_time)] = [f'{orig_language[0]}_l0_eval'] + eval_stats
                df_eval.to_excel(excel_writer, sheet_name=f'eval_stats', index=False)
                pbar_lang.n += 1
                pbar_lang.refresh()
                for i, language in enumerate(self.languages):
                    mmlu, time_stats = self.paraphrase(mmlu, language)
                    #mmlu.to_excel(excel_writer, sheet_name=f'{language[0]}_l{i+1}', index=False)
                    df_time.loc[len(df_time)] = [f'{language[0]}_l{i+1}_paraph'] + time_stats
                    df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
                    if retranslate:
                        if mid_evaluation:
                            eval, time_stats, eval_stats = self.evaluate(mmlu, language)
                            eval.to_excel(excel_writer, sheet_name=f'{language[0]}_l{i+1}', index=False)
                            df_time.loc[len(df_time)] = [f'{language[0]}_l{i+1}_eval'] + time_stats
                            df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
                            df_eval.loc[len(df_time)] = [f'{language[0]}_l0_eval'] + eval_stats
                            df_eval.to_excel(excel_writer, sheet_name=f'eval_stats', index=False)
                        mmlu2, time_stats = self.paraphrase(mmlu, orig_language)
                        df_time.loc[len(df_time)] = [f'{orig_language[0]}_l{i+1}_paraph'] + time_stats
                        df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
                        lang = orig_language
                    else:
                        mmlu2 = mmlu
                        lang = language
                    eval, time_stats, eval_stats = self.evaluate(mmlu2, lang)
                    eval.to_excel(excel_writer, sheet_name=f'{lang[0]}_l{i+1}', index=False)
                    df_time.loc[len(df_time)] = [f'{lang[0]}_l{i+1}_eval'] + time_stats
                    df_time.to_excel(excel_writer, sheet_name=f'time_stats', index=False)
                    df_eval.loc[len(df_time)] = [f'{lang[0]}_l{i+1}_eval'] + eval_stats
                    df_eval.to_excel(excel_writer, sheet_name=f'eval_stats', index=False)
                    pbar_lang.n += 1
                    pbar_lang.refresh()
            except (Exception, KeyboardInterrupt) as e:
                traceback.print_exc()
        