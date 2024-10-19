# curl -X POST http://127.0.0.1:58003/tts -F "text=Testing this inference server." -F "speaker_ref_path=https://cdn.themetavoice.xyz/speakers/bria.mp3" -F "guidance=3.0" -F "top_p=0.95" --output out.wav
import os
import logging
import shlex
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Optional
import nltk
import wave
import fastapi
import fastapi.middleware.cors
import tyro
import uvicorn
from attr import dataclass
from fastapi import File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from fastapi import Query, BackgroundTasks

from fam.llm.fast_inference import TTS
from fam.llm.utils import check_audio_file

from azure.storage.blob import ContainerClient

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
app = fastapi.FastAPI()


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

    quantisation_mode: Optional[Literal["int4", "int8"]] = None


# Singleton
class _GlobalState:
    config: ServingConfig
    tts: TTS


GlobalState = _GlobalState()

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
async def text_to_speech(
    text: str = Form(..., description="Text to convert to speech."),
    speaker_ref_path: Optional[str] = Form(None, description="Optional URL to an audio file of a reference speaker. Provide either this URL or audio data through 'audiodata'."),
    audiodata: Optional[UploadFile] = File(None, description="Optional audio data of a reference speaker. Provide either this file or a URL through 'speaker_ref_path'."),
    guidance: float = Form(3.0, description="Control speaker similarity - how closely to match speaker identity and speech style, range: 0.0 to 5.0.", ge=0.0, le=5.0),
    top_p: float = Form(0.95, description="Controls speech stability - improves text following for a challenging speaker, range: 0.0 to 1.0.", ge=0.0, le=1.0),
):
    # Ensure at least one of speaker_ref_path or audiodata is provided
    if not audiodata and not speaker_ref_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio file or a speaker reference path must be provided.",
        )

    wav_out_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_tmp:
            if speaker_ref_path is None:
                wav_path = _convert_audiodata_to_wav_path(audiodata, wav_tmp)
                check_audio_file(wav_path)
            else:
                # TODO: fix
                wav_path = speaker_ref_path

            if wav_path is None:
                warnings.warn("Running without speaker reference")
                assert guidance is None

            wav_out_path = GlobalState.tts.synthesise(
                text=text,
                spk_ref_path=wav_path,
                top_p=top_p,
                guidance_scale=guidance,
            )

        with open(wav_out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    except Exception as e:
        # traceback_str = "".join(traceback.format_tb(e.__traceback__))
        logger.exception(
            f"Error processing request. text: {text}, speaker_ref_path: {speaker_ref_path}, guidance: {guidance}, top_p: {top_p}"
        )
        return Response(
            content="Something went wrong. Please try again in a few mins or contact us on Discord",
            status_code=500,
        )
    finally:
        if wav_out_path is not None:
            Path(wav_out_path).unlink(missing_ok=True)


def _convert_audiodata_to_wav_path(audiodata: UploadFile, wav_tmp):
    with tempfile.NamedTemporaryFile() as unknown_format_tmp:
        if unknown_format_tmp.write(audiodata.read()) == 0:
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
    GlobalState.tts = TTS(
        seed=GlobalState.config.seed,
        quantisation_mode=GlobalState.config.quantisation_mode,
        telemetry_origin="api_server",
    )

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
        workers=4
    )
