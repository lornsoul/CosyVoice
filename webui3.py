# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import time
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8
# 设置说话人名称
speakers = ["穗", "安比", "派派"]
# 设置说话人信息文件的路径
model_path='pretrained_models/Fun-CosyVoice3-0.5B'
spk2info_path = f'{model_path}/spk2info.pt'
# 设置提示文本
prompt_texts = {
    "穗": "You are a helpful assistant.<|endofprompt|>我知道，那件事之后，良爷可能觉得有些事都是老天定的，人怎么做都没用，但我觉得不是这样的。",
    "安比": "You are a helpful assistant.<|endofprompt|>我在听插曲，电影里，一般不会有那么长的空镜头",
    "派派": "You are a helpful assistant.<|endofprompt|>要载你一程嘛？记得系好安全带哟",
}

# 定义一个文本到语音的函数，参数包括文本内容、是否流式处理、语速和是否使用文本前端处理
def tts_sft(tts_text, speaker, stream=False, speed=1.0, text_frontend=True):
    '''
    参数：
        tts_text：要合成的文本
        speaker：说话人音频特征
        stream：是否流式处理
        speed：语速
        text_frontend：是否使用文本前端处理

    返回值：
        合成后的音频
    '''
    speaker_info = cosyvoice.frontend.spk2info[speaker]
    # 使用tqdm库来显示进度条，对文本进行标准化处理并分割
    for i in tqdm(cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
        # 提取文本的token和长度
        tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token(i)
        # 构建模型输入字典，包括文本、文本长度、提示文本、提示文本长度、LLM提示语音token、LLM提示语音token长度、流提示语音token、流提示语音token长度、提示语音特征、提示语音特征长度、LLM嵌入和流嵌入
        prompt_token = speaker_info.get('prompt_token')
        prompt_token_len = speaker_info.get('prompt_token_len')
        speech_token = speaker_info['speech_token']
        speech_token_len = speaker_info['speech_token_len']
        speech_feat = speaker_info['speech_feat']
        speech_feat_len = speaker_info['speech_feat_len']
        embedding = speaker_info['embedding']
        model_input = {'text': tts_text_token, 
                       'text_len': tts_text_token_len,
                       'llm_prompt_speech_token': speech_token, 
                       'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token':speech_token,
                       'flow_prompt_speech_token_len':speech_token_len,
                       'prompt_speech_feat': speech_feat, 
                       'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 
                       'flow_embedding': embedding,
                       'prompt_text': prompt_token,
                       'prompt_text_len': prompt_token_len,}
        
        # 使用模型进行文本到语音的转换，并迭代输出结果
        for model_output in cosyvoice.model.tts(**model_input, stream=stream, speed=speed):
            yield model_output

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr: # type: ignore
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr)) # type: ignore
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        if cosyvoice.frontend.spk2info[sft_dropdown].get('prompt_token') is None:
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        else:
            # for i in tts_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            for i in cosyvoice.inference_zero_shot(tts_text, '', '', sft_dropdown, stream=stream, speed=speed):
                yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

def add_speaker_info(speaker, spk2info):
    if speaker in spk2info:
       del spk2info[speaker]

    if speaker not in spk2info:
        print('Extracting speaker information for {}...'.format(speaker))
        # 加载16kHz的提示语音
        prompt_wav = f'speaker_files/{speaker}.wav'
        # 获取音色embedding
        embedding = cosyvoice.frontend._extract_spk_embedding(prompt_wav)
        # 获取语音特征
        speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(prompt_wav)
        # 获取语音token
        speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(prompt_wav)
        # 提取提示文本的token和长度
        prompt_token, prompt_token_len = cosyvoice.frontend._extract_text_token(
            cosyvoice.frontend.text_normalize(prompt_texts[speaker], split=False, text_frontend=True))
        if cosyvoice.sample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        # 将音色embedding、语音特征和语音token保存到字典中
        spk2info[speaker] = {'embedding': embedding,
                        'speech_feat': speech_feat, 
                        'speech_feat_len': speech_feat_len,
                        'speech_token': speech_token,
                        'speech_token_len': speech_token_len,
                        'prompt_token': prompt_token,
                        'prompt_token_len': prompt_token_len,

                        'llm_prompt_speech_token': speech_token, 
                        'llm_prompt_speech_token_len': speech_token_len,
                        'flow_prompt_speech_token':speech_token,
                        'flow_prompt_speech_token_len':speech_token_len,
                        'prompt_speech_feat': speech_feat, 
                        'prompt_speech_feat_len': speech_feat_len,
                        'llm_embedding': embedding, 
                        'flow_embedding': embedding,
                        'prompt_text': prompt_token,
                        'prompt_text_len': prompt_token_len,}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default=model_path,
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice3(model_dir=args.model_dir)
    print(cosyvoice.frontend.spk2info.keys())

    # 记录开始时间
    start = time.time()

    # 如果说话人信息文件存在，则加载
    if os.path.exists(spk2info_path):
        spk2info = torch.load(
            spk2info_path, map_location=cosyvoice.frontend.device)
    else:
        spk2info = {}
    
    if False :
        for speaker in speakers:
            add_speaker_info(speaker, spk2info)
        torch.save(spk2info, spk2info_path)

    print('Load time:', time.time()-start)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
