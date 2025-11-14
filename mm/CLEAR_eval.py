import os
import json
import random
from tqdm import tqdm
import torch
import transformers
import hydra
from transformers import AutoProcessor, AutoConfig
from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import get_model_identifiers_from_yaml
import re
random.seed(42)


def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    reference = [ground_truth.split()]
    hypothesis = predicted_answer.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    return bleu_score


def eval_classification(model, processor, data_path, with_options, max_new_tokens):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode, with_options={with_options} #########################################")
    VQA_data = _load_clear_data(data_path)
    print(VQA_data)
    correct_count, VQA_num = 0, 0
    model_dtype = next(model.parameters()).dtype
    for idx, VQA_sample in enumerate(VQA_data):
        image = VQA_sample.get("image", None)
        question = VQA_sample.get("question", "What is the name of the person in the image?")
        answer = VQA_sample.get("name", "")
        options = VQA_sample.get("perturbed_names", [])
        options.insert(random.randint(0, len(options)), answer)
        if with_options:
            prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        else:
            prompt = question
            correct_answer = answer
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt', truncation=False).to(model.device, model_dtype)

        with torch.no_grad():
            VQA_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        out_wo_prompt = VQA_outputs[:, inputs.input_ids.shape[-1]:]
        generated_text = processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n", "**************\n", generated_text, "\n**************")

        if not with_options:
            if answer in assistant_response.lower():
                print("Correct Answer!")
                correct_count += 1
            else:
                print(f"Wrong Answer! ${assistant_response}$ doesn't include ${answer}$")
        else:
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
            if predicted_answer == correct_answer:
                print("Correct Answer!")
                correct_count += 1
            else:
                print(f"Wrong Answer! ${predicted_answer}$ != ${correct_answer}$. {answer}")
        print("##################################")
        VQA_num += 1

    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}


def eval_generation(model, processor, data_path, output_folder, mode, max_new_tokens):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode #########################################")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    df = _load_clear_data(data_path)

    results = {"Generation_Questions": []}
    avg_scores = {}

    model_dtype = next(model.parameters()).dtype
    total_bleu_VQA, total_rouge1_VQA, total_rouge2_VQA, total_rougeL_VQA, total_VQA_num = 0, 0, 0, 0, 0
    for i, VQA_sample in enumerate(tqdm(df, desc=f"{mode} VQA generation")):
        image = VQA_sample.get("image", None)
        if image is None:
            continue
        # Prefer a reasonable caption prompt if question is missing
        question = VQA_sample.get("question", "").strip()
        if not question:
            question = "Please describe the image briefly and accurately."
        # Prefer caption ground-truth if present, fallback to 'answer'
        answer = VQA_sample.get("caption", VQA_sample.get("answer", "")).strip()
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt', truncation=False).to(model.device, model_dtype)
        with torch.no_grad():
            VQA_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_wo_prompt = VQA_outputs[:, inputs.input_ids.shape[-1]:]
        generated_text = processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = generated_text
        print(f"[VQA i={i}] Generated text is : \n", "**************\n", generated_text, "\n**************")
        results["Generation_Questions"].append({
            "idx": i,
            "type": "VQA",
            "image": str(image),
            "question": question,
            "generated_answer": assistant_response,
            "ground_truth": answer
        })
        bleu_score = compute_bleu(answer, assistant_response)
        rouge_scores = rouge_scorer_obj.score(answer, assistant_response)
        total_bleu_VQA += bleu_score
        total_rouge1_VQA += rouge_scores['rouge1'].fmeasure
        total_rouge2_VQA += rouge_scores['rouge2'].fmeasure
        total_rougeL_VQA += rouge_scores['rougeL'].fmeasure
        total_VQA_num += 1

    total_bleu_QA, total_rouge1_QA, total_rouge2_QA, total_rougeL_QA, total_QA_num = 0, 0, 0, 0, 0
    for i, QA_sample in enumerate(tqdm(df, desc=f"{mode} QA generation")):
        question = QA_sample.get("question", "").strip()
        if not question:
            # skip samples without a text-only QA question
            continue
        answer = QA_sample.get("answer", "").strip()
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors='pt', truncation=False).to(model.device, model_dtype)
        with torch.no_grad():
            QA_outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_wo_prompt = QA_outputs[:, inputs.input_ids.shape[-1]:]
        generated_text = processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = generated_text
        print(f"[QA  i={i}] Generated text is : \n", "**************\n", generated_text, "\n**************")
        results["Generation_Questions"].append({
            "idx": i,
            "type": "QA",
            "image": "None",
            "question": question,
            "generated_answer": assistant_response,
            "ground_truth": answer
        })
        bleu_score = compute_bleu(answer, assistant_response)
        rouge_scores = rouge_scorer_obj.score(answer, assistant_response)
        total_bleu_QA += bleu_score
        total_rouge1_QA += rouge_scores['rouge1'].fmeasure
        total_rouge2_QA += rouge_scores['rouge2'].fmeasure
        total_rougeL_QA += rouge_scores['rougeL'].fmeasure
        total_QA_num += 1

    if total_VQA_num > 0:
        avg_scores.update({
            "Average ROUGE-1 (VQA)": total_rouge1_VQA / total_VQA_num,
            "Average ROUGE-2 (VQA)": total_rouge2_VQA / total_VQA_num,
            "Average ROUGE-L (VQA)": total_rougeL_VQA / total_VQA_num,
            "Average BLEU (VQA)": total_bleu_VQA / total_VQA_num
        })
    else:
        print(f"[{mode}] No valid VQA samples found; skipping VQA average metrics.")
        avg_scores.update({
            "Average ROUGE-1 (VQA)": "N/A",
            "Average ROUGE-2 (VQA)": "N/A",
            "Average ROUGE-L (VQA)": "N/A",
            "Average BLEU (VQA)": "N/A"
        })

    if total_QA_num > 0:
        avg_scores.update({
            "Average ROUGE-1 (QA)": total_rouge1_QA / total_QA_num,
            "Average ROUGE-2 (QA)": total_rouge2_QA / total_QA_num,
            "Average ROUGE-L (QA)": total_rougeL_QA / total_QA_num,
            "Average BLEU (QA)": total_bleu_QA / total_QA_num
        })
    else:
        print(f"[{mode}] No valid QA samples found (likely empty questions); skipping QA average metrics.")
        avg_scores.update({
            "Average ROUGE-1 (QA)": "N/A",
            "Average ROUGE-2 (QA)": "N/A",
            "Average ROUGE-L (QA)": "N/A",
            "Average BLEU (QA)": "N/A"
        })

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(f'{output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("################################## Classification Task Ends ##############################################")
    return avg_scores


def formulate_prompt_with_options(question, options, answer):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (list): The options for the question (e.g., ["Option A", "Option B"]).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    options_str = "\n".join([f"{chr(ord('A')+i)}. {value}" for i, value in enumerate(options)])
    gt = chr(ord('A') + options.index(answer))
    prompt = f"{question}\n{options_str}\n"
    return prompt, gt


def eval_classification_real(model, processor, data_path, max_new_tokens):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode #########################################")
    df = _load_clear_data(data_path)
    correct_count, VQA_num = 0, 0
    model_dtype = next(model.parameters()).dtype
    for i, sample in enumerate(df):
        question = sample.get("question", "What is the name of the person in the image?")
        answer = sample.get("answer", "")
        options = sample.get("options", [])
        options.insert(random.randint(0, len(options)), answer)
        image = sample.get("image", None)
        prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt', truncation=False).to(model.device, model_dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_wo_prompt = outputs[:, inputs.input_ids.shape[-1]:]
        generated_text = processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n", "**************\n", generated_text, "\n**************")
        predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
        if predicted_answer == correct_answer:
            print("Correct Answer!")
            correct_count += 1
        else:
            print(f"Wrong Answer! ${assistant_response}$ != ${correct_answer}$. {answer}")
        VQA_num += 1
        print("##################################")

    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}


@hydra.main(version_base=None, config_path="../config/mm", config_name="CLEAR_eval")
def main(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding_size = "longest"
        processor.do_pad = True
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_id)
    model = (
        getattr(transformers, model_cfg["hf_class"])  # trust project mapping like eval.py
        .from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    torch.cuda.empty_cache()
    results_data = {}

    # Normalize eval_list which can be a Hydra ListConfig or a string
    try:
        from omegaconf import ListConfig
        if isinstance(cfg.eval_list, (list, ListConfig)):
            eval_list = [str(e).lower() for e in list(cfg.eval_list)]
        else:
            eval_list = [str(cfg.eval_list).lower()]
    except Exception:
        eval_list = [str(cfg.eval_list).lower()] if not isinstance(cfg.eval_list, list) else [str(e).lower() for e in cfg.eval_list]
    max_new_tokens = cfg.generation.max_new_tokens

    if "forget" in eval_list:
        print("### Evaluating Forget Set ###")
        cls_path = f"{cfg.data_folder}/{cfg.forget_cls_folder}"
        gen_path = f"{cfg.data_folder}/{cfg.forget_gen_folder}"
        wo_flag = "perturbed" in cfg.forget_cls_folder.lower()
        forget_classification_result = eval_classification(
            model=model,
            processor=processor,
            data_path=cls_path,
            with_options=wo_flag,
            max_new_tokens=max_new_tokens,
        )
        forget_generation_result = eval_generation(
            model=model,
            processor=processor,
            data_path=gen_path,
            output_folder=cfg.save_dir,
            mode="forget",
            max_new_tokens=max_new_tokens,
        )
        results_data["Forget Set Results"] = {
            "classification": forget_classification_result,
            "generation": forget_generation_result,
        }

    if "retain" in eval_list:
        print("### Evaluating Retain Shared Set ###")
        cls_path = f"{cfg.data_folder}/{cfg.retain_cls_folder}"
        gen_path = f"{cfg.data_folder}/{cfg.retain_gen_folder}"
        wo_flag = "perturbed" in cfg.retain_cls_folder.lower()
        retain_classification_result = eval_classification(
            model=model,
            processor=processor,
            data_path=cls_path,
            with_options=wo_flag,
            max_new_tokens=max_new_tokens,
        )
        retain_generation_result = eval_generation(
            model=model,
            processor=processor,
            data_path=gen_path,
            output_folder=cfg.save_dir,
            mode="retain",
            max_new_tokens=max_new_tokens,
        )
        results_data["Retain Set Results"] = {
            "classification": retain_classification_result,
            "generation": retain_generation_result,
        }

    if "realface" in eval_list:
        print("### Evaluating Real Face Set ###")
        realface_classification_result = eval_classification_real(
            model=model,
            processor=processor,
            data_path=f"{cfg.data_folder}/{cfg.realface_folder}",
            max_new_tokens=max_new_tokens,
        )
        results_data["Real Face Results"] = {"classification": realface_classification_result}

    if "realworld" in eval_list:
        print("### Evaluating Real World Set ###")
        realworld_classification_result = eval_classification_real(
            model=model,
            processor=processor,
            data_path=f"{cfg.data_folder}/{cfg.realworld_folder}",
            max_new_tokens=max_new_tokens,
        )
        results_data["Real World Results"] = {"classification": realworld_classification_result}

    output_file = os.path.join(cfg.save_dir, f'{cfg.output_file}_final_evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    print(results_data)
    print(f"Results saved to {output_file}")

    with open(os.path.join(cfg.save_dir, f'{cfg.output_file}_evalconfig.json'), 'w', encoding='utf-8') as f:
        json.dump(_serialize_cfg(cfg), f, ensure_ascii=False, indent=4)


def _serialize_cfg(cfg):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg)

def _load_clear_data(path_or_repo: str):
    # Load from local disk folder saved via datasets
    if os.path.isdir(path_or_repo):
        return load_from_disk(path_or_repo)
    # Prefer interpreting suffix as dataset config (name) first, then as split
    if "/" in path_or_repo:
        base, split = path_or_repo.rsplit("/", 1)
        try:
            # Many repos use config names like 'forget10_perturbed' with split='train'
            return load_dataset(base, name=split, split="train")
        except Exception:
            try:
                # If not a config, try as a real split
                return load_dataset(base, split=split)
            except Exception:
                pass
    # Fallback: try as a full repo id with explicit split 'train'
    try:
        return load_dataset(path_or_repo, split="train")
    except Exception:
        pass
    raise ValueError(f"Could not load dataset from {path_or_repo}")


if __name__ == "__main__":
    main()