# Running Ollama without Sudo on a Remote Linux Server

This guide explains how to install and run Ollama on a Linux server where you do not have `sudo` (administrator) privileges. This is a common scenario on shared university or company servers.

## Setup

These steps only need to be done once.

1.  **Download the Ollama binary:**
    Open a terminal and download the latest Ollama executable for Linux. (Note: You can check the [Ollama GitHub releases page](https://github.com/ollama/ollama/releases) for the latest version number).
    ```bash
    curl -L https://github.com/ollama/ollama/releases/download/v0.9.3/ollama-linux-amd64 -o ollama
    ```

2.  **Make the binary executable:**
    You need to give the downloaded file permission to be run as a program.
    ```bash
    chmod +x ollama
    ```

3.  **Create a local `bin` directory and move Ollama:**
    It's good practice to keep local user programs in a `~/bin` directory.
    ```bash
    mkdir -p ~/bin
    mv ollama ~/bin
    ```

## How to Run

Because Ollama runs as a client-server application, you will typically need **two terminals**.

1.  **Terminal 1: Start the Ollama Server**
    In your first terminal, start the Ollama server. This process needs to be left running in the background to handle model requests.
    ```bash
    ~/bin/ollama serve
    ```
    You will see log messages indicating the server has started and is listening for connections. Leave this terminal window open.

2.  **Terminal 2: Run a Model**
    Open a **new, second terminal**. You can now use the `ollama` command to run a model. The first time you run a model, Ollama will download it for you, which may take some time.

    For example, to run `llama3`:
    ```bash
    ~/bin/ollama run llama3
    ```
    After the download is complete, you will see a prompt `>>>` where you can start chatting with the model.

## Usage Example

Once the model is running (you see the `>>>` prompt in your second terminal), you can type your prompts and press Enter.

```
>>> Decompose the following report into key findings, methodology, and limitations: <your report text here>
```

To exit the chat, type `/bye` and press Enter.

To stop the Ollama server completely, go back to the first terminal and press `Ctrl+C`. 



nohup bash -c 'ollama serve > output/logs/ollama.log 2>&1 & sleep 10 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct-rate && PYTHONPATH=src python src/ct_rate/report_decomposition.py' > output/logs/report_decomposition_output.log 2>&1 &

bash -c "ollama serve > output/logs/ollama.log 2>&1 & sleep 10 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct-rate && python src/ct_rate/report_decomposition.py" > output/logs/report_decomposition_output.log 2>&1