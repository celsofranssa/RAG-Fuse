import asyncio
import json
import logging
import pickle
import random
import time

import aiohttp
from omegaconf import OmegaConf
from tqdm import tqdm

from source.helper.Helper import Helper
from source.llm import llm_predict, process_llm_predict

logging.basicConfig(level=logging.INFO)


class LabelDescriptionHelper(Helper):

    def __init__(self, params):
        super(LabelDescriptionHelper, self).__init__()
        self.params = params
        self.base_url = params.llm.server.base_url
        self.model_name = params.llm.server.model
        self.prompt = self._load_prompt("optimized_prompt")
        logging.basicConfig(level=logging.INFO)

    async def _load_requests(self):
        requests = asyncio.Queue()
        with open(
                f"{self.params.llm.label_desc.request.request_dir}fold_"
                f"{self.params.llm.label_desc.request.fold_idx}/requests_{self.params.llm.label_desc.request.request_idx}"
                f".jsonl", "r") as request_file:
            for request in request_file:
                request = json.loads(request)
                request["retries"] = 0
                await requests.put(request)
        logging.info(f"Loaded {requests.qsize()} requests")
        return requests

    async def _get_batched_requests(self, requests):
        batched_requests = []
        while not requests.empty() and len(
                batched_requests) < self.params.llm.label_desc.batch_size:  # Batch size of 100
            request = await requests.get()
            batched_requests.append(request)
        return batched_requests

    def _format_labels(self, labels):
        formatted_labels = []
        for label in labels:
            formatted_labels.append(self._format_label(label))
        return '; '.join(formatted_labels)

    def _format_label(self, label):
        splited_label = label.split("->")
        if splited_label[-1] == "NA":
            return splited_label[0]
        return label

    def _get_candidates(self, samples):
        candidates = {}
        for sample in tqdm(samples, desc="Finding candidates"):
            for label_idx in sample["labels_ids"]:
                if label_idx not in candidates:
                    candidates[label_idx] = []
                candidates[label_idx].append(sample["idx"])
        return candidates

    def _get_labels_map(self, samples):
        labels_map = {}
        for sample in tqdm(samples, desc="Getting labels map"):
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels_map[label] = label_idx
        return labels_map

    def _get_text_label_pairs(self, samples, select_ids):
        text_label_pairs = ""
        for i, sample_idx in enumerate(select_ids):
            text_label_pairs += f"    text: {' '.join(samples[sample_idx]['text'].split()[:128])}\n"
            text_label_pairs += f"    labels: {self._format_labels(samples[sample_idx]['labels'])}\n\n"
        return text_label_pairs

    def _get_label_prompt(self, target_label, samples, select_ids):
        return self.prompt.format(
            target_label=target_label,
            text_label_pairs=self._get_text_label_pairs(samples, select_ids)
        )

    async def _get_requests(self, fold_idx):
        all_samples = self._load_samples()
        logging.info(f"Generating prompts in fold {fold_idx}.")
        requests = asyncio.Queue()

        num_samples = self.params.llm.label_desc.num_samples  # 5
        samples = self._load_split_samples(fold_idx, "train") + self._load_split_samples(fold_idx, "val")

        labels_map = self._get_labels_map(samples)
        candidates = self._get_candidates(samples)

        for target_label, label_idx in tqdm(labels_map.items(), desc="Describing labels"):
            target_label = self._format_label(target_label)
            samples_ids = candidates[label_idx]
            select_ids = samples_ids if len(samples_ids) < num_samples else random.sample(samples_ids, num_samples)
            prompt = self._get_label_prompt(target_label, all_samples, select_ids)

            await requests.put(
                {
                    "prompt": prompt,
                    "label_idx": label_idx,
                    "body": {
                        "prompt": prompt,
                        "max_gen_len": self.params.llm.prompt_opt.max_gen_len,
                        "temperature": self.params.llm.prompt_opt.temperature,
                        "top_p": self.params.llm.prompt_opt.top_p
                    },
                    "model": self.model_name,
                    "timeout": self.params.llm.prompt_opt.timeout
                }
                #     {
                #     "recordId": f"label_{label_idx}",
                #     "modelInput": {
                #         "prompt": prompt,
                #         "max_gen_len": self.params.llm.label_desc.max_gen_len,
                #         "temperature": self.params.llm.label_desc.temperature,
                #         "top_p": self.params.llm.label_desc.top_p}
                # }
            )

        return requests

    async def _process_requests(self, prompts_requests):
        labels_descriptions = {}
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
                            labels_descriptions[request["label_idx"]] = request["response"]

                    pbar.update(len(batched_requests) - num_failures)

        return labels_descriptions

    async def _run(self):
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Describing labels {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")
            requests = await self._get_requests(fold_idx)
            labels_descriptions = await self._process_requests(requests)
            self._checkpoint_label_descriptions(labels_descriptions, fold_idx)

    def run(self):
        asyncio.run(
            self._run()
        )

    def _checkpoint_label_descriptions(self, labels_descriptions, fold_idx):
        logging.info(
            f"Checkpointing {len(labels_descriptions)} labels on {self.params.data.dir}/fold_{fold_idx}/labels_descriptions.pkl")
        with open(f"{self.params.data.dir}fold_{fold_idx}/labels_descriptions.pkl", "wb") as labels_desc_file:
            pickle.dump(labels_descriptions, labels_desc_file)
