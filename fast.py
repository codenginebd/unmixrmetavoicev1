import json
import logging
import shlex
import subprocess
import tempfile
import warnings
import nltk
import wave
from pathlib import Path
from typing import Optional
import torch
import os
import datetime
from fastapi import FastAPI, Query, BackgroundTasks
import fastapi.middleware.cors
import tyro
import uvicorn
from azure.storage.blob import ContainerClient
from attr import dataclass
from fastapi import Request
from fastapi.responses import Response

from fam.llm.fast_inference import TTS
from fam.llm.utils import check_audio_file

logger = logging.getLogger(__name__)


def azure_initiate(
        result_blob: str,
        storage_connection_string: str,
):
    azure_client = ContainerClient.from_connection_string(
        storage_connection_string, result_blob
    )
    return azure_client


def retrieve_file(container_client, file_name):
    return container_client.get_blob_client(file_name)


def split_into_sentences(text):
    nltk.download('punkt')  # Download the Punkt tokenizer.
    return nltk.tokenize.sent_tokenize(text)


def combine_wav_files(input_files, output_file):
    data = []

    for file in input_files:
        with wave.open(file, 'rb') as wav_file:
            data.append([wav_file.getparams(), wav_file.readframes(wav_file.getnframes())])

    output_params = data[0][0]

    with wave.open(output_file, 'wb') as output_wav_file:
        output_wav_file.setparams(output_params)
        for params, frames in data:
            output_wav_file.writeframes(frames)


# create local paths
output_dir = '.data/outputs'
voice_dir = '.data/input_voice'
text_dir = '.data/input_text'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(voice_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)
os.makedirs('.data/tmp', exist_ok=True)
## Setup FastAPI server.
app = FastAPI()


def inference(connection_string: str, input_container_name: str, output_container_name: str, voices_container_name: str,
              reference_voice: str, text_file: str = None):
    # authentiacate in azure

    input_blob = azure_initiate(input_container_name, connection_string)
    result_blob = azure_initiate(output_container_name, connection_string)
    voices_blob = azure_initiate(voices_container_name, connection_string)

    process_start_time = datetime.datetime.now()

    # start processing

    # Load the data (voice reference and text)
    if text_file not in os.listdir(text_dir):
        # download blob with name text_file from input_container_name
        text = retrieve_file(input_blob, text_file).download_blob().readall()
        with open(f"{text_dir}/{text_file}", "wb") as f:
            f.write(text)
    # read the text file
    with open(f"{text_dir}/{text_file}", "r") as f:
        text = f.read()
    # download reference voice
    voice_local_path = f'{voice_dir}/{reference_voice}'
    if reference_voice not in os.listdir(voice_dir):
        voice = retrieve_file(voices_blob, reference_voice).download_blob()
        with open(f"{voice_local_path}", "wb") as f:
            voice.readinto(f)

    # TTS process:

    tts_req = TTSRequest(text=text, speaker_ref_path=voice_local_path)
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
        if tts_req.speaker_ref_path is None:
            wav_path = "./assets/bria.mp3"
        else:
            # TODO: fix
            wav_path = tts_req.speaker_ref_path

        if wav_path is None:
            warnings.warn("Running without speaker reference")
            assert tts_req.guidance is None
        if len(tts_req.text.split()) > 10:
            sentences = split_into_sentences(tts_req.text)
            list_of_wav_out = []
            for sentence in sentences:
                wav_out_path = GlobalState.tts.synthesise(
                    text=sentence,
                    spk_ref_path=wav_path,
                    top_p=tts_req.top_p,
                    guidance_scale=tts_req.guidance,
                )
                list_of_wav_out.append(wav_out_path)
            wav_out_path = "." + text_file.split(".")[1] + "_" + reference_voice.split(".")[0] + "_meta" + ".wav"
            combine_wav_files(list_of_wav_out, wav_out_path)
        else:
            wav_out_path = GlobalState.tts.synthesise(
                text=tts_req.text,
                spk_ref_path=wav_path,
                top_p=tts_req.top_p,
                guidance_scale=tts_req.guidance,
            )
    # save result to azure
    end_time = datetime.datetime.now()
    result_file_name = text_file.split(".")[0] + "_" + reference_voice.split(".")[0] + "_meta" + ".wav"
    output_blob_client = result_blob.get_blob_client(result_file_name)
    # upload wav file from local path to blob
    with open(wav_out_path, "rb") as bytes_data:
        output_blob_client.upload_blob(bytes_data, overwrite=True)

    return {"status": "success", "result saved to": f"{output_container_name}/{result_file_name}",
            "processing time": str(end_time - process_start_time)}


@dataclass
class ServingConfig:
    huggingface_repo_id: str = "metavoiceio/metavoice-1B-v0.1"
    """Absolute path to the model directory."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    port: int = 58003


# Singleton
class _GlobalState:
    def __init__(self):
        self.config = ServingConfig()  # Initialize config
        self.tts = TTS()  # Initialize tts


GlobalState = _GlobalState()


# GlobalState.config = tyro.cli(ServingConfig)
# GlobalState.tts = TTS(seed=GlobalState.config.seed)

@dataclass(frozen=True)
class TTSRequest:
    text: str
    speaker_ref_path: Optional[str] = None
    guidance: float = 3.0
    top_p: float = 0.95
    top_k: Optional[int] = None


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/process_short_text")
async def process(
        connection_string: str = Query(
            "DefaultEndpointsProtocol=https;AccountName=accountname;AccountKey=key;EndpointSuffix=core.windows.net",
            description="Azure Storage Connection String"),
        input_container_name: str = Query("requests", description="Container name for input files"),
        output_container_name: str = Query("results", description="Container name for output files"),
        voices_container_name: Optional[str] = Query("voices", description="Container name for voice files"),
        reference_voice: Optional[str] = Query(None, description="Voice file to be used as reference"),
        text_file: str = Query(description="Text file to be used for TTS"),
):
    result = inference(connection_string, input_container_name, output_container_name, voices_container_name,
                       reference_voice, text_file)
    return result


@app.post("/process_long_text")
async def process_long(
        background_tasks: BackgroundTasks,
        connection_string: str = Query(
            "DefaultEndpointsProtocol=https;AccountName=accountname;AccountKey=key;EndpointSuffix=core.windows.net",
            description="Azure Storage Connection String"),
        input_container_name: str = Query("requests", description="Container name for input files"),
        output_container_name: str = Query("results", description="Container name for output files"),
        voices_container_name: Optional[str] = Query("voices", description="Container name for voice files"),
        reference_voice: Optional[str] = Query(None, description="Voice file to be used as reference"),
        text_file: str = Query(description="Text file to be used for TTS"),
):
    background_tasks.add_task(inference, connection_string, input_container_name, output_container_name,
                              voices_container_name, reference_voice, text_file)
    return {"status": "Process started"}


@app.post("/tts", response_class=Response)
async def text_to_speech(req: Request):
    audiodata = await req.body()
    payload = None
    wav_out_path = None

    try:
        headers = req.headers
        payload = headers["X-Payload"]
        payload = json.loads(payload)
        tts_req = TTSRequest(**payload)
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
            if tts_req.speaker_ref_path is None:
                wav_path = _convert_audiodata_to_wav_path(audiodata, wav_tmp)
                check_audio_file(wav_path)
            else:
                # TODO: fix
                wav_path = tts_req.speaker_ref_path

            if wav_path is None:
                warnings.warn("Running without speaker reference")
                assert tts_req.guidance is None

            wav_out_path = GlobalState.tts.synthesise(
                text=tts_req.text,
                spk_ref_path=wav_path,
                top_p=tts_req.top_p,
                guidance_scale=tts_req.guidance,
            )

        with open(wav_out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    except Exception as e:
        # traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logger.exception(f"Error processing request {payload}")
        return Response(
            content="Something went wrong. Please try again in a few mins or contact us on Discord",
            status_code=500,
        )
    finally:
        if wav_out_path is not None:
            Path(wav_out_path).unlink(missing_ok=True)


def _convert_audiodata_to_wav_path(audiodata, wav_tmp):
    with tempfile.NamedTemporaryFile() as unknown_format_tmp:
        if unknown_format_tmp.write(audiodata) == 0:
            return None
        unknown_format_tmp.flush()

        subprocess.check_output(
            # arbitrary 2 minute cutoff
            shlex.split(f"ffmpeg -t 120 -y -i {unknown_format_tmp.name} -f wav {wav_tmp.name}")
        )

        return wav_tmp.name


if __name__ == "__main__":
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)

    GlobalState.config = tyro.cli(ServingConfig)
    GlobalState.tts = TTS(seed=GlobalState.config.seed)

    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*", f"http://localhost:{GlobalState.config.port}", "http://localhost:80"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(
        app,
        host="::",
        port=GlobalState.config.port,
        log_level="info",
        workers=8
    )