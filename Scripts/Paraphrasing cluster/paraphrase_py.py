import argparse, os, gc, time, traceback



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
		"--cuda",
		type=str,
        default="0",
		required=False,
		help='CUDA_VISIBLE_DEVICES',
	)

    parser.add_argument(
        "-i",
		"--input",
		type=str,
        default="preguntas_extensas_MMLU3.xlsx",
		required=False,
		help="input's path",
	)
    
    parser.add_argument(
        "-o",
		"--output",
		type=str,
        default="results_10_iterations_3-6",
		required=False,
		help="output's folder path",
	)

    parser.add_argument(
        "-f",
		"--format",
		type=str,
        choices=["all", "csv", "xlsx"],
        default="all",
		required=False,
		help="output's saving format",
	)

    parser.add_argument(
        "-m",
		"--model",
		type=str,
		required=False,
        default="meta-llama/Llama-3.1-8B-Instruct",
		help="model's name",
	)

    parser.add_argument(
        "-b",
		"--batchsize",
		type=int,
        default=10,
		required=False,
		help="how many iterations is the save performed",
	)

    parser.add_argument(
        "-s",
		"--save",
		type=int,
        default=1000,
		required=False,
		help="how many iterations is the save performed",
	)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from autoasker import AutoAskerException, AutoAskerTranslator, torch
    from tqdm import tqdm



    models = [
                {"model_id":args.model, "b_params":20, "default_config":False, "batch_size":args.batchsize},
    ]

    prompts = [
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>",
        "The following JSON contains a multiple choice question. Please rephrase the values while maintaining the original meaning to the same language. Ensure that the question is presented differently but conveys the same idea. Keep the JSON format in the answer with '{' and '}'.\n<json>. Please provide only the requested JSON with no additional text and do a good paraphrase, being the length of the paraphrase of at least of 500 characters.",
        "Answer the following multiple choice question. Keep the answer in JSON format and stick to the following form '{answer: option key}' with only one key and value.\n<json>"
    ]
    
    languages = [
            [('en','english'), ('en','english'),('en','english'), ('en','english')],

    ]

    extra_names = ["_EN"]
    
    
    for i in tqdm(range(len(models)), desc='Models completed', colour='yellow'):
        m = models[i]['model_id'].split(sep='/')[1]
        a = None
        extra_name = extra_names[i]
        try:
            a = AutoAskerTranslator(args.input, models[i], prompts, languages[i])
            a.run(args.output, retranslate = False, mid_evaluation = False, lines = 0, extra_name=extra_name)
        except (Exception, KeyboardInterrupt) as e:
            traceback.print_exc()
            if a:
                print("\nSaving data and progress...")
                #a.save_progress()
                #a.generate_result('csv')
                print("Saved!")
        finally:
            del a
            gc.collect()
            torch.cuda.empty_cache()
    time.sleep(10)