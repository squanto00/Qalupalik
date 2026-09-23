# Qalupalik

Qalupalik is a personal AI model project built around a small, locally runnable language model.

The goal is to develop **Qalupalik as a base AI**, with different versions specialized for different purposes.

Rather than building one model that tries to do everything, Qalupalik is intended to serve as a model family:

```text
Qalupalik
├── Base
├── D&D
├── Coding
├── Writing
└── Experimental
```

Individual versions can have their own training, instruction tuning, tools, personalities, and purposes while sharing the same underlying Qalupalik foundation.

## Current Base

The current foundation is **OpenAI GPT-2 Small (124M)**.

Architecture:

* 124,439,808 parameters
* 12 transformer layers
* 12 attention heads
* 768-dimensional embeddings
* 1,024-token context
* 50,257-token vocabulary

The original TensorFlow checkpoint has been converted to modern PyTorch/SafeTensors format.

## Project Status

* [x] Obtain original GPT-2 124M checkpoint
* [x] Set up modern Python environment
* [x] Set up CPU-only PyTorch
* [x] Set up modern Transformers
* [x] Inspect original TensorFlow checkpoint
* [x] Convert checkpoint to SafeTensors
* [ ] Verify text generation
* [ ] Establish Qalupalik base model workflow
* [ ] Create specialized Qalupalik variants
* [ ] Develop training/instruction-tuning pipeline
* [ ] Add tools and external capabilities
* [ ] Build model versioning system

## Model Variants

Specialized versions of Qalupalik will be developed independently depending on their intended purpose.

Examples:

* **Qalupalik-DM**: tabletop RPG / D&D assistant
* **Qalupalik-Coder**: programming and technical tasks
* **Qalupalik-Writer**: creative writing and storytelling
* **Qalupalik-Experimental**: testing new architectures, datasets, and ideas

These names are examples of the intended model-family structure and are not necessarily existing models yet.

## Philosophy

Qalupalik is an experiment in building useful local AI from a deliberately small foundation.

The project focuses on:

* Local execution
* Open tooling
* Modifiability
* Specialized model variants
* Understanding how the system works rather than treating the model as a black box

The goal is not simply to run an existing model.

The goal is to **build the system around it**.
