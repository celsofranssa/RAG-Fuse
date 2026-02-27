import asyncio
import logging
import random
import re

import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from source.helper.Helper import Helper
from source.llm import llm_predict, process_llm_predict


def _extract_prompt(response):
    pattern = r'<prompt>(.*?)<\/prompt>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def _is_valid_prompt(prompt, required_vars):
    missing_vars = [var for var in required_vars if var not in prompt]
    if missing_vars and prompt != "":
        return False
    return True


class PromptOptimizerHelper(Helper):
    def __init__(self, params):
        super(PromptOptimizerHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.base_url = params.llm.server.base_url
        self.model_name = params.llm.server.model
        self.samples = self._load_samples()
        logging.info(f"Loaded {len(self.samples)} samples")

    def _get_candidates(self):
        candidates = {}
        for sample in tqdm(self.samples, desc="Finding candidates"):
            for label in sample["labels"]:
                if label not in candidates:
                    candidates[label] = []
                candidates[label].append(sample["idx"])
        return candidates

    def _format_label(self, label):
        splited_label = label.split("->")
        if splited_label[-1] == "NA":
            return splited_label[0]
        return label

    def _format_labels(self, labels):
        formatted_labels = []
        for label in labels:
            formatted_labels.append(self._format_label(label))
        return '; '.join(formatted_labels)

    def _get_text_label_pairs(self, select_ids):
        text_label_pairs = ""
        for _, sample_idx in enumerate(select_ids):
            text_label_pairs += f"    text: {' '.join(self.samples[sample_idx]['text'].split()[:128])}\n"
            text_label_pairs += f"    labels: {self._format_labels(self.samples[sample_idx]['labels'])}\n\n"
        return text_label_pairs

    def _get_prompt_samples(self, target_descriptions, candidates):
        prompt_samples = []
        for target_label, target_description in target_descriptions.items():
            for _ in range(self.params.llm.prompt_opt.num_samples_per_target_label):
                selected_ids = random.choices(candidates[target_label], k=3)
                prompt_samples.append({
                    "text_label_pairs": self._get_text_label_pairs(selected_ids),
                    "target_label": target_label,
                    "target_description": target_description
                })
        return prompt_samples

    def run(self):
        asyncio.run(
            self._run()
        )

    async def _run(self):
        seed_prompt = self._load_prompt("seed_prompt")
        meta_prompt = self._load_prompt("meta_prompt")
        target_descriptions = self._load_target_descriptions()
        candidates = self._get_candidates()
        prompt_samples = self._get_prompt_samples(target_descriptions, candidates)

        optimized_prompts = await self._optimize_prompt(meta_prompt, [(seed_prompt, 0.0)], prompt_samples)

        optimized_prompt = max(optimized_prompts, key=lambda x: x[1])[0]

        self._checkpoint_prompt(optimized_prompt, "optimized_prompt")

    async def _get_batched_requests(self, requests):
        batched_requests = []
        while not requests.empty() and len(batched_requests) < self.params.llm.prompt_opt.batch_size:
            request = await requests.get()
            batched_requests.append(request)
        return batched_requests

    async def _optimize_prompt(self, meta_prompt, prompts, prompt_samples):

        for epoch in range(self.params.llm.prompt_opt.num_epochs):

            # select the best description prompt until now
            current_description_prompt = max(prompts, key=lambda x: x[1])[0]

            # update the currentta prompt
            # select top k (k=2) top prompts by its scores
            prompts_scores = "\n\n".join(
                [f"Prompt:\n<prompt>\n{prompt}\n</prompt>\nScore: {score}" for prompt, score in prompts[:2]]
            )

            formated_meta_prompt = meta_prompt.format(
                prompts_scores=prompts_scores,
                description_prompt=current_description_prompt,
                prompt_samples=self._get_sample_description_prompt(prompt_samples, current_description_prompt)
            )

            current_description_prompt = await self._predict_prompt(formated_meta_prompt,
                                                                    stopping_criterion=["</prompt>"])
            current_description_prompt = _extract_prompt(current_description_prompt)

            # while not a valide prompt, try again
            while not _is_valid_prompt(current_description_prompt,
                                       required_vars=["text_label_pairs", "{target_label}"]):
                current_description_prompt = await self._predict_prompt(formated_meta_prompt,
                                                                        stopping_criterion=["</prompt>"])
                current_description_prompt = _extract_prompt(current_description_prompt)

            # submit current_description_prompt with data
            prompts_requests = asyncio.Queue()
            for sample in tqdm(prompt_samples, desc=f"Epoch {epoch}"):
                prompt = current_description_prompt.format(
                    text_label_pairs=sample["text_label_pairs"],
                    target_label=sample["target_label"]
                )
                await prompts_requests.put({
                    "prompt": prompt,
                    "target_description": sample["target_description"],
                    "body": {
                        "prompt": prompt,
                        "max_gen_len": self.params.llm.prompt_opt.max_gen_len,
                        "temperature": self.params.llm.prompt_opt.temperature,
                        "top_p": self.params.llm.prompt_opt.top_p
                    },
                    "model": self.model_name,
                    "timeout": self.params.llm.prompt_opt.timeout
                })

            pred_descriptions, true_descriptions = [], []
            async with aiohttp.ClientSession() as http_session:
                with tqdm(total=prompts_requests.qsize(), desc=f"Requesting") as pbar:
                    while not prompts_requests.empty():  # Process the queue in batches until its empty
                        batched_requests = await self._get_batched_requests(prompts_requests)
                        processed_requests = await process_llm_predict(
                            http_session,
                            self.base_url,
                            batched_requests
                        )
                        num_failures = 0
                        for request in processed_requests:
                            if request["status"] == "failure":
                                num_failures += 1
                                await prompts_requests.put(request)
                            else:
                                pred_descriptions.append(request["response"])
                                true_descriptions.append(request["target_description"])

                        pbar.update(len(batched_requests) - num_failures)

            # eval effectiveness
            score = self.get_effectiveness(true_descriptions, pred_descriptions)
            print(f"Effectiveness: {score}")
            prompts.append(
                (current_description_prompt, score)
            )

        return prompts

    def _get_sample_description_prompt(self, prompt_samples, description_prompt):
        sample = random.choice(prompt_samples)
        prompt = description_prompt.format(
            target_label=sample["target_label"],
            text_label_pairs=sample["text_label_pairs"]
        )
        return prompt

    async def _predict_prompt(self, prompt, stopping_criterion):
        prompt_request = {
            "body": {
                "prompt": prompt,
                "max_gen_len": self.params.llm.prompt_opt.max_gen_len,
                "temperature": self.params.llm.prompt_opt.temperature,
                "top_p": self.params.llm.prompt_opt.top_p,
                "stop": stopping_criterion
            },
            "model": self.model_name,
            "timeout": self.params.llm.prompt_opt.timeout,
            "status": ""
        }

        async with aiohttp.ClientSession() as http_session:
            while prompt_request["status"] != "success":
                prompt_request = await llm_predict(
                    http_session,
                    self.base_url,
                    prompt_request
                )
            return prompt_request["response"] + "</prompt>"

    def get_effectiveness(self, true_descriptions, pred_descriptions):
        embeddings1 = self.embedding_model.encode(true_descriptions, convert_to_numpy=True)
        embeddings2 = self.embedding_model.encode(pred_descriptions, convert_to_numpy=True)
        similarities = np.diag(cosine_similarity(embeddings1, embeddings2))
        return np.mean(similarities)
