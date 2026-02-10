import os
import sys
from dotenv import load_dotenv
import shutil

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing
import warnings
import yaml

warnings.simplefilter("ignore")

from tqdm import tqdm
from modules.commons import *
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from hf_utils import load_custom_model_from_hf

import os
import sys
import torch
import time
import numpy as np
from modules.commons import str2bool

# 全局变量初始化
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

flag_vc = False
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds
fp16 = False

# 性能指标全局变量
first_packet_latency = 0.0  # 首包延迟(ms)
total_inference_time = 0.0  # 总推理时间(s)
audio_total_duration = 0.0   # 音频总时长(s)
processed_chunks = 0         # 处理的块数

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0,
                 ):
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    
    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    
    # 更新参考音频相关参数
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")
    
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    
    # 语义特征提取计时
    if device.type == "mps":
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        torch.mps.synchronize()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        #torch.cuda.synchronize()

    # 推理核心计时
    infer_start = time.time()
    #start_event.record()
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    #end_event.record()
    
    if device.type == "mps":
        torch.mps.synchronize()
    #else:
        #torch.cuda.synchronize()
    #elapsed_time_ms = start_event.elapsed_time(end_event)
    #print(f"Semantic feature extract time: {elapsed_time_ms:.2f}ms")

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    
    cond = model.length_regulator(
        S_alt, ylens=target_lengths , n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target).squeeze()
    
    infer_end = time.time()
    infer_duration = infer_end - infer_start
    
    # 更新全局性能指标
    global first_packet_latency, total_inference_time, processed_chunks
    if processed_chunks == 0:  # 首包延迟
        first_packet_latency = infer_duration * 1000
    total_inference_time += infer_duration
    processed_chunks += 1

    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

def load_models(args):
    global fp16
    fp16 = args.fp16
    print(f"Using fp16: {fp16}")
    
    # 加载模型配置和权重
    if args.checkpoint_path is None or args.checkpoint_path == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_uvit_tat_xlsr_ema.pth",
                                                                         "config_dit_mel_seed_uvit_xlsr_tiny.yml")
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path
        
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # 加载模型权重
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # 加载CAMPPlus
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # 加载Vocoder
    vocoder_type = model_params.vocoder.type
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    # 加载语音Tokenizer
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    
    # 构建Mel频谱函数
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

def stream_audio_inference(args):
    """
    流式音频推理主函数：读取WAV文件 → 分块推理 → 流式输出保存
    """
    global audio_total_duration
    
    # 1. 加载模型
    print("Loading models...")
    model_set = load_models(args)
    sr = model_set[5]["sampling_rate"]  # 获取模型采样率
    sr_16k = 16000
    
    # 2. 加载参考音频和输入音频
    if not os.path.exists(args.reference_audio):
        raise FileNotFoundError(f"Reference audio not found: {args.reference_audio}")
    if not os.path.exists(args.input_audio):
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")
    
    # 加载参考音频
    ref_audio, _ = librosa.load(args.reference_audio, sr=sr)
    ref_audio_name = os.path.basename(args.reference_audio)
    
    # 加载输入音频（流式处理的源音频）
    input_audio, _ = librosa.load(args.input_audio, sr=sr)
    audio_total_duration = len(input_audio) / sr
    print(f"Input audio loaded. Duration: {audio_total_duration:.2f}s, Sample rate: {sr}")
    
    # 3. 配置流式参数
    block_time = args.block_time  # 每个块的时间长度(s)
    block_samples = int(block_time * sr)  # 每个块的采样点数
    crossfade_samples = int(args.crossfade_time * sr)  # 交叉淡化采样点数
    max_prompt_length = args.max_prompt_length
    diffusion_steps = args.diffusion_steps
    inference_cfg_rate = args.inference_cfg_rate
    extra_time_ce = args.extra_time_ce
    extra_time_right = args.extra_time_right
    
    # 计算块相关参数（16k采样率下）
    block_frame_16k = int(block_time * sr_16k)
    skip_head = int(extra_time_ce * 50)  # 50 = 16000/320 (whisper stride)
    skip_tail = int(extra_time_right * 50)
    return_length = int(block_time * 50)
    
    # 4. 初始化输出缓冲区和保存路径
    output_audio_chunks = []
    output_dir = "stream_output"
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, f"output_{os.path.basename(args.input_audio)}")
    
    # 5. 流式处理音频
    print("\nStarting streaming inference...")
    start_time = time.time()
    current_pos = 0
    
    with tqdm(total=int(audio_total_duration / block_time), desc="Processing chunks") as pbar:
        while current_pos < len(input_audio):
            # 切分当前块（处理最后一块不足的情况）
            end_pos = min(current_pos + block_samples, len(input_audio))
            audio_chunk = input_audio[current_pos:end_pos]
            
            # 补零（最后一块）
            if len(audio_chunk) < block_samples:
                audio_chunk = np.pad(audio_chunk, (0, block_samples - len(audio_chunk)), mode='constant')
            
            # 转换为16k采样率（模型输入要求）
            audio_chunk_tensor = torch.from_numpy(audio_chunk).to(device)
            audio_chunk_16k = torchaudio.functional.resample(audio_chunk_tensor, sr, sr_16k)
            
            # 执行推理
            output_wave = custom_infer(
                model_set=model_set,
                reference_wav=ref_audio,
                new_reference_wav_name=ref_audio_name,
                input_wav_res=audio_chunk_16k,
                block_frame_16k=block_frame_16k,
                skip_head=skip_head,
                skip_tail=skip_tail,
                return_length=return_length,
                diffusion_steps=diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                max_prompt_length=max_prompt_length,
                cd_difference=extra_time_ce
            )
            
            # 转换为numpy并添加到输出缓冲区（处理交叉淡化）
            output_wave_np = output_wave.cpu().numpy()
            if len(output_audio_chunks) > 0:
                # 交叉淡化处理
                prev_chunk = output_audio_chunks[-1]
                overlap = min(crossfade_samples, len(prev_chunk), len(output_wave_np))
                if overlap > 0:
                    # 淡出前一块，淡入当前块
                    fade_out = np.linspace(1.0, 0.0, overlap)
                    fade_in = np.linspace(0.0, 1.0, overlap)
                    prev_chunk[-overlap:] = prev_chunk[-overlap:] * fade_out
                    output_wave_np[:overlap] = output_wave_np[:overlap] * fade_in
                    output_wave_np[:overlap] += prev_chunk[-overlap:]
            
            output_audio_chunks.append(output_wave_np)
            
            # 更新进度
            current_pos += block_samples
            pbar.update(1)
    
    # 6. 合并输出并保存WAV
    total_time = time.time() - start_time
    global total_inference_time
    total_inference_time = total_time
    
    # 合并所有块并裁剪补零部分
    output_audio = np.concatenate(output_audio_chunks)
    output_audio = output_audio[:int(audio_total_duration * sr)]  # 裁剪到原音频长度
    
    # 保存为WAV文件
    torchaudio.save(
        final_output_path,
        torch.from_numpy(output_audio).unsqueeze(0).float(),
        sr
    )
    print(f"\nOutput saved to: {final_output_path}")
    
    # 7. 计算并打印性能指标
    print("\n===== Performance Metrics =====")
    print(f"First packet latency: {first_packet_latency:.2f} ms")
    print(f"Total inference time: {total_inference_time:.2f} s")
    print(f"Audio total duration: {audio_total_duration:.2f} s")
    print(f"Processed chunks: {processed_chunks}")
    print(f"Real-Time Factor (RTF): {total_inference_time / audio_total_duration:.4f}")  # RTF<1表示实时
    print(f"Average chunk processing time: {total_inference_time / processed_chunks * 1000:.2f} ms/chunk")

class Config:
    def __init__(self):
        self.device = device

if __name__ == "__main__":
    import argparse
    
    # 构建命令行参数
    parser = argparse.ArgumentParser(description="Seed-VC Streaming Inference from WAV File")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference WAV audio file")
    parser.add_argument("--input-audio", type=str, required=True, help="Path to input WAV audio file")
    parser.add_argument("--checkpoint-path", type=str, default="./DiT_uvit_tat_xlsr_ema.pth", help="Custom model checkpoint path")
    parser.add_argument("--config-path", type=str, default="./config_dit_mel_seed_uvit_xlsr_tiny.yml", help="Custom model config path")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use FP16 inference")
    parser.add_argument("--block-time", type=float, default=0.25, help="Block time in seconds (default: 0.25)")
    parser.add_argument("--crossfade-time", type=float, default=0.04, help="Crossfade time in seconds (default: 0.04)")
    parser.add_argument("--extra-time-ce", type=float, default=2.5, help="Extra context left (default: 2.5)")
    parser.add_argument("--extra-time-right", type=float, default=0.02, help="Extra context right (default: 0.02)")
    parser.add_argument("--diffusion-steps", type=int, default=10, help="Diffusion steps (default: 10)")
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7, help="Inference CFG rate (default: 0.7)")
    parser.add_argument("--max-prompt-length", type=float, default=3.0, help="Max prompt length in seconds (default: 3.0)")
    
    args = parser.parse_args()
    
    # 执行流式推理
    try:
        stream_audio_inference(args)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
