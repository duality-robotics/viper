# ViperGPT for VQA

This repository presents an enhanced version of the original [ViperGPT: Visual Inference via Python Execution for Reasoning paper implementation](https://github.com/cvlab-columbia/viper), introducing significant updates to leverage the capabilities of GPT-4 and advanced vision models for Visual Question Answering (VQA).

## Introduction

ViperGPT introduced an innovative approach to VQA, utilizing a large language model (LLM) in conjunction with multiple vision models. The key strategy is to separate visual processing tasks from general reasoning, allowing for modular updates and transparent decision-making. Our enhancements include updates to the llm_query and code generation functions, incorporation of the Segment Anything Model (SAM) for masking, and updates to the vision models for improved performance.

## Table of Contents

- [Try it yourself](#try-it-yourself)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)

## Try it yourself
![Downtown Visual Reasoning](./assets/FalconVisualReasoning.gif)

You can experience the power of ViperGPT for VQA in a photorealistic simulation firsthand by following these steps:

1. **Create a Falcon Account:** Sign up for a free account on [Falcon](https://falcon.duality.ai/).

2. **Run the Downtown Visual Reasoning scenario:** Once logged in, navigate to the [Downtown Visual Reasoning scenario](https://falcon.duality.ai/secure/scenarios/edit/ba748712-d5cc-4fe8-8ca7-42a8938dc9c0) and run it directly in your browser.

3. **Explore the Environment:** Use the controls to navigate the mannequin robot around the outdoor environment. Observe how ViperGPT handles various visual reasoning tasks in this simulated setting.

Feel free to experiment and explore different prompts to see how ViperGPT performs as the visual system of an embodied AI system.


## Installation
    
Follow these steps to install the ViperGPT framework:

1. **Clone recursively:**
    ```bash
    git clone --recurse-submodules https://github.com/duality-robotics/viper.git
    ```

2. **cd into the viper directory:**
    ```bash
    cd viper
    ```

2. **OpenAI key:** To run the GPT-4, you will need to configure an OpenAI key. This can be done by signing up for an account e.g. here, and then creating a key in account/api-keys. Create a file api.key in the root of this project and store the key in it.
    ```
    echo YOUR_OPENAI_API_KEY_HERE > api.key
    ```

3. **Create a virtual environment:**
   ```bash
   python -m venv vipervenv
   ```

4. **Activate the virtual environment:**
   ```bash
   source ./viperenv/bin/activate
   ```
5. **Install PyTorch:** Visit [PyTorch's official site](https://pytorch.org/get-started/locally/) and follow the instructions based on your CUDA version.

6. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
7. **Build GroundingDINO:**
    - Change to the GroundingDINO folder:
        ```bash
        cd GroundingDINO
        ```
    - Install the required dependencies in the current directory:
        ```bash
        pip install -e .
        ```
    - Download pre-trained model weights:
        ```bash
        mkdir weights
        cd weights
        curl -LO https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
        cd ../..
        ```

## Usage

To use ViperGPT for VQA tasks, follow the documentation provided in the `viper_gpt.ipynb` notebook. This notebook includes detailed instructions on setting up the environment, loading models, and executing queries.

## Features

- **Modular Framework:** Easily replace visual or reasoning models to incorporate the latest advancements.
- **GPT-4 Integration:** Utilizes GPT-4 for advanced reasoning and natural language processing.
- **Vision Model Updates:** Incorporates state-of-the-art vision models for various tasks, such as GroundingDINO, BLIP2, and the addition of SAM for masking.
- **Photorealistic Simulation:** Includes a setup in a Falcon-made photorealistic simulation for embodied AI testing.

| Function/property name | Original ViperGPT model | Our implementation model |
|------------------------|------------------------|--------------------------|
| find                   | GLIP                   | GroundingDINO            |
| exists                 | GLIP                   | BLIP2                    |
| verify_property        | X-VLM                  | BLIP2                    |
| simple_query           | BLIP2                  | BLIP2                    |
| best_image_match       | X-VLM                  | BLIP2                    |
| best_text_match        | CLIP                   | N/A                      |
| compute_depth          | MiDaS                  | MiDaS                    |
| masks                  | N/A                    | SAM                      |
| llm_query              | GPT-3                  | GPT-4                    |
| code generation        | Codex                  | GPT-4                    |

## Dependencies

The project depends on several python libraries, including:
- openai
- transformers
- accelerate
- bitsandbytes
- GroundingDINO

Ensure these are installed using the `pip install -r requirements.txt` command during installation.

## Configuration

Refer to the `viper_context.py` and `image_patch.py` scripts for configuration options related to model parameters and image processing.

## Examples

For examples of how to use the ViperGPT framework, see the `viper_gpt.ipynb` notebook. It provides sample code for initializing the framework, processing images, and querying the LLM.
