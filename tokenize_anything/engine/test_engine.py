# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Engine for testing."""

import time

from tokenize_anything.models.easy_build import model_registry


class InferenceCommand(object):
    """Command to run batched inference."""

    def __init__(self, input_queue, output_queue, kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.kwargs = kwargs

    def build_env(self):
        """Build the environment."""
        self.batch_size = self.kwargs.get("batch_size", 1)
        self.batch_timeout = self.kwargs.get("batch_timeout", None)

    def build_model(self):
        """Build and return the model."""
        builder = model_registry[self.kwargs["model_type"]]
        return builder(device=self.kwargs["device"], checkpoint=self.kwargs["weights"])

    def build_predictor(self, model):
        """Build and return the predictor."""
        return self.kwargs["predictor_type"](model, self.kwargs)

    def send_results(self, predictor, indices, examples):
        """Send the inference results."""
        results = predictor.get_results(examples)
        if hasattr(predictor, "timers"):
            time_diffs = dict((k, v.average_time) for k, v in predictor.timers.items())
            for i, outputs in enumerate(results):
                self.output_queue.put((indices[i], time_diffs, outputs))
        else:
            for i, outputs in enumerate(results):
                self.output_queue.put((indices[i], outputs))

    def run(self):
        """Main loop to make the inference outputs."""
        self.build_env()
        model = self.build_model()
        predictor = self.build_predictor(model)
        must_stop = False
        while not must_stop:
            indices, examples = [], []
            deadline, timeout = None, None
            for i in range(self.batch_size):
                if self.batch_timeout and i == 1:
                    deadline = time.monotonic() + self.batch_timeout
                if self.batch_timeout and i >= 1:
                    timeout = deadline - time.monotonic()
                try:
                    index, example = self.input_queue.get(timeout=timeout)
                    if index < 0:
                        must_stop = True
                        break
                    indices.append(index)
                    examples.append(example)
                except Exception:
                    pass
            if len(examples) == 0:
                continue
            self.send_results(predictor, indices, examples)
