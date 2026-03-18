import os
import sys
import argparse
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

def create_frontend(model_dir):
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    # 1. 侦测 CosyVoice 版本 (通过yaml文件)
    if os.path.exists(os.path.join(model_dir, 'cosyvoice3.yaml')):
        yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        tokenizer_name = 'speech_tokenizer_v3.onnx'
        version = 3
    elif os.path.exists(os.path.join(model_dir, 'cosyvoice2.yaml')):
        yaml_path = os.path.join(model_dir, 'cosyvoice2.yaml')
        tokenizer_name = 'speech_tokenizer_v2.onnx'
        version = 2
    elif os.path.exists(os.path.join(model_dir, 'cosyvoice.yaml')):
        yaml_path = os.path.join(model_dir, 'cosyvoice.yaml')
        tokenizer_name = 'speech_tokenizer_v1.onnx'
        version = 1
    else:
        raise ValueError(f"No valid cozyvoice yaml config found in {model_dir}")

    # 2. 解析 configurations
    with open(yaml_path, 'r') as f:
        if version == 3:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        else:
            configs = load_hyperpyyaml(f)
            
    # 3. 实例化 FrontEnd
    tokenizer_path = os.path.join(model_dir, tokenizer_name)
    if not os.path.exists(tokenizer_path):
        # 回退机制
        tokenizer_path = tokenizer_path.replace('.onnx', '.pt') 

    frontend = CosyVoiceFrontEnd(
        configs['get_tokenizer'],
        configs['feat_extractor'],
        os.path.join(model_dir, 'campplus.onnx'),
        tokenizer_path,
        os.path.join(model_dir, 'spk2info.pt'),
        configs.get('allowed_special', 'all')
    )
    
    return frontend, configs['sample_rate']


def load_spk2info(spk2info_path, device='cpu'):
    if os.path.exists(spk2info_path):
        return torch.load(spk2info_path, map_location=device)
    return {}


def save_spk2info(spk2info, spk2info_path):
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(spk2info_path)), exist_ok=True)
    torch.save(spk2info, spk2info_path)
    print(f"Saved speaker info to {spk2info_path}")


def list_speakers(args):
    spk2info_path = os.path.join(args.model_dir, 'spk2info.pt')
    spk2info = load_spk2info(spk2info_path)
    speakers = list(spk2info.keys())
    print(f"Total {len(speakers)} speakers found in {spk2info_path}:")
    for spk in speakers:
        print(f"  - {spk}")


def remove_speaker(args):
    spk2info_path = os.path.join(args.model_dir, 'spk2info.pt')
    spk2info = load_spk2info(spk2info_path)
    if args.name in spk2info:
        del spk2info[args.name]
        save_spk2info(spk2info, spk2info_path)
        print(f"Successfully removed speaker '{args.name}'.")
    else:
        print(f"Speaker '{args.name}' not found.")


def add_speaker(args):
    spk2info_path = os.path.join(args.model_dir, 'spk2info.pt')

    # Load frontend only
    print(f"Loading CosyVoice Frontend from {args.model_dir}...")
    frontend, sample_rate = create_frontend(args.model_dir)

    spk2info = load_spk2info(spk2info_path, device=frontend.device)

    speaker = args.name
    print(f"Extracting speaker information for {speaker}...")

    prompt_wav = args.wav
    prompt_text = args.text

    if not os.path.exists(prompt_wav):
        print(f"Error: Prompt wav file '{prompt_wav}' not found.")
        return

    embedding = frontend._extract_spk_embedding(prompt_wav)
    speech_feat, speech_feat_len = frontend._extract_speech_feat(prompt_wav)
    speech_token, speech_token_len = frontend._extract_speech_token(prompt_wav)
    prompt_token, prompt_token_len = frontend._extract_text_token(
        frontend.text_normalize(prompt_text, split=False, text_frontend=True))

    if sample_rate == 24000:
        # cosyvoice2, force speech_feat % speech_token = 2
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

    spk2info[speaker] = {
        'embedding': embedding,
        'speech_feat': speech_feat,
        'speech_feat_len': speech_feat_len,
        'speech_token': speech_token,
        'speech_token_len': speech_token_len,
        'prompt_token': prompt_token,
        'prompt_token_len': prompt_token_len,

        'llm_prompt_speech_token': speech_token,
        'llm_prompt_speech_token_len': speech_token_len,
        'flow_prompt_speech_token': speech_token,
        'flow_prompt_speech_token_len': speech_token_len,
        'prompt_speech_feat': speech_feat,
        'prompt_speech_feat_len': speech_feat_len,
        'llm_embedding': embedding,
        'flow_embedding': embedding,
        'prompt_text': prompt_token,
        'prompt_text_len': prompt_token_len,
    }

    save_spk2info(spk2info, spk2info_path)
    print(f"Successfully added speaker '{speaker}'.")


def main():
    parser = argparse.ArgumentParser(description="CosyVoice Speaker Info Manager")
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='Directory of the model containing spk2info.pt')

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # list command
    parser_list = subparsers.add_parser('list', help='List all available speakers')
    parser_list.set_defaults(func=list_speakers)

    # add command
    parser_add = subparsers.add_parser('add', help='Add a new speaker')
    parser_add.add_argument('--name', type=str, required=True, help='Name of the speaker')
    parser_add.add_argument('--wav', type=str, required=True, help='Path to the prompt audio file (16kHz recommended)')
    parser_add.add_argument('--text', type=str, required=True, help='Text corresponding to the prompt audio')
    parser_add.set_defaults(func=add_speaker)

    # remove command
    parser_remove = subparsers.add_parser('remove', help='Remove a speaker')
    parser_remove.add_argument('--name', type=str, required=True, help='Name of the speaker to remove')
    parser_remove.set_defaults(func=remove_speaker)

    args = parser.parse_args()
    args.func(args)

# 设置提示文本
prompt_texts = {
    "穗": "You are a helpful assistant.<|endofprompt|>我知道，那件事之后，良爷可能觉得有些事都是老天定的，人怎么做都没用，但我觉得不是这样的。",
    "安比": "You are a helpful assistant.<|endofprompt|>我在听插曲，电影里，一般不会有那么长的空镜头",
    "派派": "You are a helpful assistant.<|endofprompt|>要载你一程嘛？记得系好安全带哟",
		"柚木": "You are a helpful assistant.<|endofprompt|>アラッ！メガサメタノネ…！タイチョウハ…ドウカシラ？",
}

if __name__ == '__main__':
    main()
