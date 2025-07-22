SCRIPT_NAME    = "DaVinci Whisper"
SCRIPT_VERSION = " 1.0"
SCRIPT_AUTHOR  = "HEIBA"

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
WINDOW_WIDTH, WINDOW_HEIGHT = 300, 350
X_CENTER = (SCREEN_WIDTH  - WINDOW_WIDTH ) // 2
Y_CENTER = (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2

SCRIPT_KOFI_URL      = "https://ko-fi.com/heiba"
SCRIPT_BILIBILI_URL  = "https://space.bilibili.com/385619394"
LANGUAGE_MAP = {
    "Auto":None,
    "中文（普通话）": "zh",
    "中文（粤语）": "yue",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "Spanish": "es",
    "Portuguese": "pt",
    "French": "fr",
    "Indonesian": "id",
    "German": "de",
    "Russian": "ru",
    "Italian": "it",
    "Arabic": "ar",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
    "Uzbek": "uz",
    "Dutch": "nl"
}

import os
import time
import platform
import sys
import random
import webbrowser
import string
import shutil
import glob
import re
from typing import Optional, List, Generator, Dict, Callable
from abc import ABC, abstractmethod

SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
AUDIO_TEMP_DIR = os.path.join(SCRIPT_PATH, "audio_temp")
SUB_TEMP_DIR = os.path.join(SCRIPT_PATH, "sub_temp")
RAND_CODE = "".join(random.choices(string.digits, k=2))

ui       = fusion.UIManager
dispatcher = bmd.UIDispatcher(ui)
loading_win = dispatcher.AddWindow(
    {
        "ID": "LoadingWin",
        "WindowTitle": "Loading",
        "Geometry": [X_CENTER, Y_CENTER, WINDOW_WIDTH, WINDOW_HEIGHT],
        "Spacing": 10,
        "StyleSheet": "*{font-size:14px;}"
    },
    [
        ui.VGroup(
            [
                ui.Label(
                    {
                        "ID": "LoadLabel",
                        "Text": "Loading...",
                        "Alignment": {"AlignHCenter": True, "AlignVCenter": True},
                    }
                )
            ]
        )
    ]
)
loading_win.Show()

# ================== DaVinci Resolve Connection ==================
try:
    import DaVinciResolveScript as dvr_script
    from python_get_resolve import GetResolve
    print("DaVinciResolveScript from Python")
except ImportError:
    if platform.system() == "Darwin":
        path1 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Examples"
        path2 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
    elif platform.system() == "Windows":
        path1 = os.path.join(os.environ['PROGRAMDATA'], "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Examples")
        path2 = os.path.join(os.environ['PROGRAMDATA'], "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Modules")
    else:
        raise EnvironmentError("Unsupported operating system")
    sys.path += [path1, path2]
    import DaVinciResolveScript as dvr_script
    from python_get_resolve import GetResolve
    print("DaVinciResolveScript from DaVinci")

def connect_resolve():
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    timeline = project.GetCurrentTimeline()
    fps = float(project.GetSetting("timelineFrameRate"))
    return resolve, project, media_pool, root_folder, timeline, fps

if not hasattr(sys.stderr, "flush"):
    sys.stderr.flush = lambda: None

try:
    import faster_whisper
except ImportError:
    system = platform.system()
    if system == "Windows":
        lib_dir = os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), "Blackmagic Design", "DaVinci Resolve", "Fusion", "HB", SCRIPT_NAME, "Lib")
    elif system == "Darwin":
        lib_dir = os.path.join("/Library", "Application Support", "Blackmagic Design", "DaVinci Resolve", "Fusion", "HB", SCRIPT_NAME, "Lib")
    else:
        lib_dir = os.path.normpath(os.path.join(SCRIPT_PATH, "..", "..", "..", "HB", SCRIPT_NAME, "Lib"))

    lib_dir = os.path.normpath(lib_dir)
    if os.path.isdir(lib_dir):
        sys.path.insert(0, lib_dir)
    else:
        print(f"Warning: The Whisper/Lib directory doesn’t exist:{lib_dir}", file=sys.stderr)

    try:
        import faster_whisper
        print(lib_dir)
    except ImportError as e:
        print("Dependency import failed—please make sure all dependencies are bundled into the Lib directory:", lib_dir, "\nError message:", e)

# ================== Transcription Provider Abstraction ==================

class TranscriptionProvider(ABC):
    """Abstract base class for transcription services."""
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Returns a list of available model names."""
        pass

    @abstractmethod
    def transcribe(self, **kwargs) -> Optional[str]:
        """
        Performs transcription and returns the path to the generated SRT file,
        or None on failure.
        """
        pass

class FasterWhisperProvider(TranscriptionProvider):
    """Transcription provider using the faster-whisper library."""

    def get_available_models(self) -> List[str]:
        return ["tiny", "small", "base", "medium", "large-v3"]

    def _format_time(self, seconds: float) -> str:
        milliseconds = int((seconds % 1) * 1000)
        return time.strftime('%H:%M:%S', time.gmtime(seconds)) + f',{milliseconds:03d}'

    def _split_segments_by_max_chars(self, segments: Generator, max_chars: int) -> List[Dict]:
        END_OF_CLAUSE_CHARS = tuple(".,?!。，？！")
        subtitle_blocks = []
        current_block = {"start": 0, "end": 0, "text": ""}
        max_chars_tolerance = int(max_chars * 1.20)

        def finalize_and_reset_block():
            nonlocal current_block
            if current_block["text"]:
                subtitle_blocks.append(current_block)
            current_block = {"start": 0, "end": 0, "text": ""}

        for segment in segments:
            if not segment.words:
                continue
            for word in segment.words:
                word_text = word.word
                if not current_block["text"]:
                    current_block = {"start": word.start, "end": word.end, "text": word_text.lstrip()}
                    continue

                potential_len = len(current_block["text"]) + len(word_text)
                word_ends_clause = word_text.strip().endswith(END_OF_CLAUSE_CHARS)

                if potential_len <= max_chars:
                    current_block["text"] += word_text
                    current_block["end"] = word.end
                    if word_ends_clause:
                        finalize_and_reset_block()
                elif potential_len <= max_chars_tolerance and word_ends_clause:
                    current_block["text"] += word_text
                    current_block["end"] = word.end
                    finalize_and_reset_block()
                else:
                    finalize_and_reset_block()
                    current_block = {"start": word.start, "end": word.end, "text": word_text.lstrip()}
        
        finalize_and_reset_block()
        return subtitle_blocks

    def _progress_reporter(self, segments_gen, total_duration: float, callback, max_fps: float = 10.0):
        if total_duration <= 0:
            for seg in segments_gen: yield seg
            callback(100.0)
            return

        last_end, last_report_ts = 0.0, 0.0
        min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        progress = 0.0

        for seg in segments_gen:
            last_end = max(last_end, seg.end)
            progress = min(last_end / total_duration * 100.0, 100.0)
            now = time.time()
            if now - last_report_ts >= min_interval or progress >= 100.0:
                last_report_ts = now
                callback(progress)
            yield seg
        
        if progress < 100.0:
            callback(100.0)

    def _remove_gaps_between_blocks(self, blocks: List[Dict]) -> List[Dict]:
        if len(blocks) < 2:
            return blocks
        for i in range(len(blocks) - 1):
            blocks[i]["end"] = blocks[i+1]["start"]
        return blocks

    def transcribe(self, **kwargs) -> Optional[str]:
        input_audio = kwargs.get("input_audio")
        model_name = kwargs.get("model_name", "base")
        language = kwargs.get("language")
        output_dir = kwargs.get("output_dir", ".")
        output_filename = kwargs.get("output_filename")
        max_chars = kwargs.get("max_chars", 40)
        batch_size = kwargs.get("batch_size", 4)
        hotwords = kwargs.get("hotwords")
        verbose = kwargs.get("verbose", True)
        progress_callback = kwargs.get("progress_callback")
        vad_filter = kwargs.get("vad_filter", False)
        remove_gaps = kwargs.get("remove_gaps", False)

        local_model_path = os.path.join(SCRIPT_PATH, "model", model_name)
        try:
            if verbose: 
                show_dynamic_message(f"Loading model '{model_name}'...", f"正在加载模型 '{model_name}'...")
            model = faster_whisper.WhisperModel(local_model_path)
            pipeline = faster_whisper.BatchedInferencePipeline(model=model)
            if verbose: 
                show_dynamic_message(f"Model '{model_name}' loaded.", f"模型 '{model_name}' 加载成功。")
        except Exception as e:
            show_dynamic_message(f"Model '{model_name}' is unavailable", f"模型'{model_name}'不可用")
            print(f"Error loading model {model_name}: {e}")
            return None

        transcribe_args = {"beam_size": 5, "log_progress": True, "batch_size": batch_size, "word_timestamps": True, "hotwords": hotwords, "vad_filter": vad_filter}
        print(transcribe_args)
        if language:
            transcribe_args["language"] = language
        
        if verbose: 
            show_dynamic_message("[Whisper] Starting...", "[Whisper] 开始...")
        segments_gen, info = pipeline.transcribe(input_audio, **transcribe_args)
        if verbose: 
            show_dynamic_message(f"[Whisper] Language: {info.language}", f"[Whisper] 语言: {info.language}")

        if progress_callback:
            segments_gen = self._progress_reporter(segments_gen, info.duration, progress_callback)
        
        if info.language in ['zh', 'ja', 'th', 'lo', 'km', 'my', 'bo']:
            max_chars = max_chars / 2
        subtitle_blocks = self._split_segments_by_max_chars(segments_gen, int(max_chars))
        
        if remove_gaps:
            subtitle_blocks = self._remove_gaps_between_blocks(subtitle_blocks)
        
        for blk in subtitle_blocks:
            blk["text"] = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", blk["text"])
        
        if not output_filename:
            base = os.path.splitext(os.path.basename(input_audio))[0]
            output_filename = f"{base}_whisper"
        os.makedirs(output_dir, exist_ok=True)
        srt_path = os.path.join(output_dir, f"{output_filename}.srt")

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, blk in enumerate(subtitle_blocks, 1):
                f.write(f"{idx}\n")
                f.write(f"{self._format_time(blk['start'])} --> {self._format_time(blk['end'])}\n")
                f.write(f"{blk['text'].strip()}\n\n")

        if verbose: print(f"[Whisper] Generated SRT: {srt_path}")
        return srt_path

# ================== UI Definition and Logic ==================

# Instantiate the desired transcription provider
faster_whisper_provider = FasterWhisperProvider()

win = dispatcher.AddWindow(
    {
        "ID": 'MyWin',
        "WindowTitle": SCRIPT_NAME + SCRIPT_VERSION,
        "Geometry": [X_CENTER, Y_CENTER, WINDOW_WIDTH, WINDOW_HEIGHT],
        "Spacing": 10,
        "StyleSheet": "*{font-size:14px;}"
    },
    [
        ui.VGroup([
            ui.VGroup({"Weight":1},[
                ui.Label({"ID":"TitleLabel","Text":"Create subtitles from audio","Alignment": {"AlignHCenter": True, "AlignVCenter": True},"Weight":0.1}),
                ui.HGroup({"Weight":0.1},[
                    ui.Label({"ID":"ModelLabel","Text":"Model","Weight":0.1}),
                    ui.ComboBox({"ID":"ModelCombo","Weight":0.1}),
                ]),
                ui.HGroup({"Weight":0.1},[
                    ui.Label({"ID":"LangLabel","Text":"Language","Weight":0.1}),
                    ui.ComboBox({"ID":"LangCombo","Weight":0.1}),
                ]),
                
                ui.HGroup({"Weight":0.1},[
                    ui.Label({"ID":"MaxCharsLabel","Text":"Max Chars","Weight":0.1}),
                    ui.SpinBox({"ID": "MaxChars", "Minimum": 0, "Maximum": 100, "Value": 42, "SingleStep": 1, "Weight": 0.1}),
                ]),
                ui.CheckBox({"ID":"NoGapCheckBox", "Text":"No Gaps Between Subtitles", "Checked":False, "Weight":0}),
                ui.Button({"ID":"CreateButton","Text":"Create","Weight":0.15}),
                ui.Label({"ID":"HotwordsLabel","Text":"Phrases","Weight":0.1}),
                ui.TextEdit({"ID":"Hotwords","Text":"","Weight":0.1}),
                ui.HGroup({"Weight":0.1},[
                    ui.CheckBox({"ID":"LangEnCheckBox","Text":"EN","Checked":True,"Weight":0}),
                    ui.CheckBox({"ID":"LangCnCheckBox","Text":"简体中文","Checked":False,"Weight":0}),
                ]),
                ui.Button({
                        "ID": "CopyrightButton", 
                        "Text": f"© 2025, Copyright by {SCRIPT_AUTHOR}",
                        "Alignment": {"AlignHCenter": True, "AlignVCenter": True},
                        "Font": ui.Font({"PixelSize": 12, "StyleName": "Bold"}),
                        "Flat": True,
                        "TextColor": [0.1, 0.3, 0.9, 1],
                        "BackgroundColor": [1, 1, 1, 0],
                        "Weight": 0
                    })
            ]),     
        ])
    ])
msgbox = dispatcher.AddWindow(
        {
            "ID": 'msg',
            "WindowTitle": 'Warning',
            "Geometry": [750, 400, 350, 100],
            "Spacing": 10,
        },
        [
            ui.VGroup(
                [
                    ui.Label({"ID": 'WarningLabel', "Text": ""}),
                    ui.HGroup({"Weight": 0}, [ui.Button({"ID": 'OkButton', "Text": 'OK'})]),
                ]
            ),
        ]
    )

def show_dynamic_message(en_text, zh_text):
    use_en = items["LangEnCheckBox"].Checked
    msg = en_text if use_en else zh_text
    msgbox.Show()
    msg_items["WarningLabel"].Text = msg

def on_msg_close(ev):
    msgbox.Hide()
msgbox.On.OkButton.Clicked = on_msg_close
msgbox.On.msg.Close = on_msg_close

translations = {
    "cn": {
        "TitleLabel":"从音频创建字幕", 
        "LangLabel":"语言", 
        "ModelLabel":"模型", 
        "CreateButton":"创建", 
        "HotwordsLabel":"短语列表", 
        "MaxCharsLabel":"每行最大字符", 
        "NoGapCheckBox":"字幕之间无间隙"},
    "en": {
        "TitleLabel":"Create subtitles from audio", 
        "LangLabel":"Language", 
        "ModelLabel":"Model", 
        "CreateButton":"Create", 
        "HotwordsLabel":"Phrases", 
        "MaxCharsLabel":"Max Chars", 
        "NoGapCheckBox":"No Gaps Between Subtitles"}
}

items = win.GetItems()
msg_items = msgbox.GetItems()

for lang_display_name in LANGUAGE_MAP.keys():
    items["LangCombo"].AddItem(lang_display_name)

# Populate models from the provider
for model in faster_whisper_provider.get_available_models():
    items["ModelCombo"].AddItem(model)

def switch_language(lang):
    for item_id, text_value in translations[lang].items():
        if item_id in items:
            items[item_id].Text = text_value

def on_lang_checkbox_clicked(ev):
    is_en_checked = ev['sender'].ID == "LangEnCheckBox"
    items["LangCnCheckBox"].Checked = not is_en_checked
    items["LangEnCheckBox"].Checked = is_en_checked
    switch_language("en" if is_en_checked else "cn")

win.On.LangCnCheckBox.Clicked = on_lang_checkbox_clicked
win.On.LangEnCheckBox.Clicked = on_lang_checkbox_clicked

def import_srt_to_first_empty(path):
    resolve, current_project, current_media_pool, current_root_folder, current_timeline, fps = connect_resolve()
    if not current_timeline:
        return False

    states = {}
    for i in range(1, current_timeline.GetTrackCount("subtitle") + 1):
        states[i] = current_timeline.GetIsTrackEnabled("subtitle", i)
        if states[i]:
            current_timeline.SetTrackEnable("subtitle", i, False)

    target = next(
        (i for i in range(1, current_timeline.GetTrackCount("subtitle") + 1)
         if not current_timeline.GetItemListInTrack("subtitle", i)),
        None
    )
    if target is None:
        current_timeline.AddTrack("subtitle")
        target = current_timeline.GetTrackCount("subtitle")
    current_timeline.SetTrackEnable("subtitle", target, True)

    srt_folder = None
    for folder in current_root_folder.GetSubFolderList():
        if folder.GetName() == "srt":
            srt_folder = folder
            break
    if srt_folder is None:
        srt_folder = current_media_pool.AddSubFolder(current_root_folder, "srt")

    current_media_pool.SetCurrentFolder(srt_folder)

    current_media_pool.ImportMedia([path])

    clips = srt_folder.GetClipList()
    latest_clip = clips[-1]  
    current_media_pool.AppendToTimeline([latest_clip])

    print("🎉 The subtitles were inserted into folder 'srt' and track #", target)
    return True

def load_audio_only_preset(project, keyword="audio only"):
    presets = project.GetRenderPresetList() or []
    def norm(x): return (x if isinstance(x, str) else x.get("PresetName","")).lower()
    hit = next((p for p in presets if keyword in norm(p)), None)
    if hit:
        name = hit if isinstance(hit, str) else hit.get("PresetName")
        if project.LoadRenderPreset(name): return name
    if project.LoadRenderPreset("Audio Only"): return "Audio Only"
    return None

def render_timeline_audio(output_dir: str, custom_name: str) -> Optional[str]:
    resolve, project, _, _, timeline, _ = connect_resolve()
    if not project or not timeline: return None
    
    load_audio_only_preset(project)
    os.makedirs(output_dir, exist_ok=True)
    render_settings = {
        "SelectAllFrames": True, 
        "ExportVideo": False, 
        "ExportAudio": True,
        "TargetDir": output_dir, 
        "CustomName": custom_name,
        "AudioSampleRate": 48000, 
        "AudioCodec": "LinearPCM", 
        "AudioBitDepth": 16,
    }
    project.SetRenderSettings(render_settings)

    job_id = project.AddRenderJob()
    print(f"Render job added, ID: {job_id}")

    if not job_id: 
        return None
    
    project.StartRendering([job_id], isInteractiveMode=False)
    print("Rendering in progress, waiting for completion...")
    while project.IsRenderingInProgress():
        print("Rendering...")
        time.sleep(2)
        
    print("Render complete!")
    return os.path.join(output_dir, f"{custom_name}.wav")

def on_create_clicked(ev):
    resolve, _, _, _, timeline, _ = connect_resolve()
    if not timeline:
        show_dynamic_message("No active timeline.", "没有激活的时间线。")
        return
        
    timeline_name = timeline.GetName()
    safe_name = timeline_name.replace(" ", "_")
    audio_file_prefix = f"{safe_name}_audio_temp"
    audio_path = os.path.join(AUDIO_TEMP_DIR, f"{audio_file_prefix}.wav")
    
    raw_hotwords = items["Hotwords"].PlainText or ""
    hotwords_list = [
        ph.strip() 
        for ph in re.split(r"[，,、；;]\s*|\s+", raw_hotwords) 
        if ph.strip()
    ]
    
    def update_transcribe_progress(progress):
        show_dynamic_message(f"Transcribing... {progress:.1f}%", f"转录中... {progress:.1f}%")
    
    try:
        show_dynamic_message("Checking for cached audio...", "检查音频缓存...")
        print(f"Checking for existing audio file with prefix '{audio_file_prefix}'")
        if not os.path.exists(audio_path):
            show_dynamic_message("Rendering audio...", "音频处理中...")
            audio_path = render_timeline_audio(output_dir=AUDIO_TEMP_DIR, custom_name=audio_file_prefix)
        else:
            print(f"Found cached audio: {audio_path}. Skipping render.")

        if not audio_path:
            show_dynamic_message("Failed to get audio file.", "获取音频文件失败。")
            return

        pattern = os.path.join(SUB_TEMP_DIR, f"{timeline_name}_subtitle_*.srt")
        indices = [int(f.split('_')[-1].split('.')[0]) for f in glob.glob(pattern) if f.split('_')[-1].split('.')[0].isdigit()]
        next_idx = max(indices) + 1 if indices else 1
        filename = f"{timeline_name}_subtitle_{RAND_CODE}_{next_idx}"

        show_dynamic_message("Transcribing... 0.0%", "转录中... 0.0%")
        resolve.OpenPage("edit")
        
        srt_path = faster_whisper_provider.transcribe(
            input_audio=audio_path,
            model_name=items["ModelCombo"].CurrentText,
            language=LANGUAGE_MAP.get(items["LangCombo"].CurrentText),
            output_dir=SUB_TEMP_DIR,
            output_filename=filename,
            max_chars=items["MaxChars"].Value,
            batch_size=4,
            hotwords=",".join(hotwords_list) if hotwords_list else None,
            vad_filter=True,
            progress_callback=update_transcribe_progress,
            remove_gaps=items["NoGapCheckBox"].Checked
        )
        
        if srt_path:
            import_srt_to_first_empty(srt_path)
            show_dynamic_message("Finished! 100%", "转录完成！")
        else:
            print("Failed to generate SRT. Model loading might have failed.")
            
    except Exception as e:
        show_dynamic_message(f"Error: {e}", f"错误: {e}")
        print(f"An error occurred: {e}")
        
win.On.CreateButton.Clicked = on_create_clicked

def on_open_link_button_clicked(ev):
    url = SCRIPT_KOFI_URL if items["LangEnCheckBox"].Checked else SCRIPT_BILIBILI_URL
    webbrowser.open(url)
win.On.CopyrightButton.Clicked = on_open_link_button_clicked

def on_close(ev):
    for temp_dir in [AUDIO_TEMP_DIR, SUB_TEMP_DIR]:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except OSError as e:
                print(f"Error removing directory {temp_dir}: {e.strerror}")
    dispatcher.ExitLoop()
win.On.MyWin.Close = on_close

loading_win.Hide() 
win.Show()
dispatcher.RunLoop()
win.Hide()