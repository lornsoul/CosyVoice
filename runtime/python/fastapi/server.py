# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import torchaudio
import io

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
        yield tts_audio


def generate_wav_response(model_output):
    tts_audio = b""
    for i in model_output:
        tts_audio += (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
    tts_speech = torch.from_numpy(
        np.array(np.frombuffer(tts_audio, dtype=np.int16))
    ).unsqueeze(dim=0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, tts_speech, 22050, format="wav")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="audio.wav"'},
    )


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_sft_wav")
@app.post("/inference_sft_wav")
async def inference_sft_wav(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return generate_wav_response(model_output)


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot_wav")
@app.post("/inference_zero_shot_wav")
async def inference_zero_shot_wav(
    tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k
    )
    return generate_wav_response(model_output)


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual_wav")
@app.post("/inference_cross_lingual_wav")
async def inference_cross_lingual_wav(
    tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return generate_wav_response(model_output)


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()
):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct_wav")
@app.post("/inference_instruct_wav")
async def inference_instruct_wav(
    tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()
):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return generate_wav_response(model_output)


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2_wav")
@app.post("/inference_instruct2_wav")
async def inference_instruct2_wav(
    tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
    )
    return generate_wav_response(model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../../../pretrained_models/CosyVoice2-0.5B",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
