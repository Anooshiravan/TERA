# TERA: Transformer Endpoint Rest API
This project is a collection of python scripts and Jupyter notebooks for working with Generative Pretrained Transformers (GPT) models provided by the EleutherAI and HuggingFace AI community. GPT is a neural network machine learning model trained using internet data to generate any type of text.


[![HuggingFace](IMG/hflogo.png)](https://huggingface.co/EleutherAI/)

API is tested with the following models:
* EleutherAI/gpt-neo-125M
* EleutherAI/gpt-neo-1.3B
* EleutherAI/gpt-neo-2.7B
* EleutherAI/gpt-j-6B


# | Server
It is a Python Uvicorn server, based on FastAPI, to provide a Rest API endpoint to EleutherAI GPT models.
It detects the CUDA configuration on the laptop and will load GPT-Neo-125m (125 million parameters) for development instead of GPT-J-6B (6 billion parameters)
The smaller model is sufficient for test/development. However, due to much smaller learning data in comparison with GPT-J, the conversation is clearly at a lower NLP level.


## Install on a Windows laptop:
If you have an nVidia GPU on the system (e.g. laptop) install latest version of Nvidia drivers.  
After reboot, device manager shows the available graphic card. 
To ensure that Nvidia drivers are working properly and the GPU is recognized, use the **nvidia-smi** command line utility:  

![](IMG/nvidia-smi.png)

The minimum version of Python to run the Hugging Face transformers for GPT-J-6B is Python 3.10.x  
PyTorch needs to be installed by the command provided by PyTorch website:
https://pytorch.org/get-started/locally/

![](IMG/pytorch.png)

The installation command is:
```pip install torch torchvision torchaudio — extra-index-url https://download.pytorch.org/whl/cu113```

To install Jupyter notebook, the Hugging Face transformers and FastAPI, try:
```pip install — upgrade pip```
```pip install jupyter```
```pip install transformers```
```pip install fastapi```
```pip install "uvicorn[standard]"```

At this point you must be able to run server by going to SERVER folder and execute:
```uvicorn main:app```

The output looks like this:  

![](IMG/server.png)

The URL to browse the API endpoint:
http://localhost:8000/docs

To test the endpoint, expand the POST section and click on "Try it out" button.
Change the tokens to e.g. 256 and type a string for prompt:  

![](IMG/prompt.png)

After clicking on "Execute" botton, the output is generated:  

![](IMG/response.png)

If a response (generated text) received from API, it means the Tera server is configured properly.  



