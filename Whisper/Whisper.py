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
from typing import Optional, List, Generator,Dict

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
        "Geometry": [X_CENTER, Y_CENTER, WINDOW_WIDTH, WINDOW_HEIGHT],                  # [x, y, width, height]
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

# ================== DaVinci Resolve 接入 ==================
try:
    import DaVinciResolveScript as dvr_script
    from python_get_resolve import GetResolve
    print("DaVinciResolveScript from Python")
except ImportError:
    # mac / windows 常规路径补全
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
    media_pool = project.GetMediaPool(); 
    root_folder = media_pool.GetRootFolder()
    timeline      = project.GetCurrentTimeline()
    fps     = float(project.GetSetting("timelineFrameRate"))
    return resolve, project, media_pool,root_folder,timeline, fps

if not hasattr(sys.stderr, "flush"):
    sys.stderr.flush = lambda: None

try:
    import faster_whisper
except ImportError:
    system = platform.system()
    if system == "Windows":
        program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        lib_dir = os.path.join(
            program_data,
            "Blackmagic Design",
            "DaVinci Resolve",
            "Fusion",
            "HB",
            SCRIPT_NAME,
            "Lib"
        )
    elif system == "Darwin":
        lib_dir = os.path.join(
            "/Library",
            "Application Support",
            "Blackmagic Design",
            "DaVinci Resolve",
            "Fusion",
            "HB",
            SCRIPT_NAME,
            "Lib"
        )
    else:
        lib_dir = os.path.normpath(
            os.path.join(SCRIPT_PATH, "..", "..", "..", "HB", SCRIPT_NAME, "Lib")
        )

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
                    #ui.CheckBox({"ID":"VADCheckBox","Text":"VAD","Checked":False,"Weight":0}),
                    ui.CheckBox({"ID":"NoGapCheckBox", "Text":"No Gaps Between Subtitles", "Checked":False, "Weight":0}),
                    ui.Button({"ID":"CreateButton","Text":"Create","Weight":0.15}),
                    #ui.Label({"ID": "StatusLabel", "Text": " ", "Alignment": {"AlignHCenter": True, "AlignVCenter": True},"Weight":0.1}),
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
                    ui.HGroup(
                        {
                            "Weight": 0,
                        },
                        [
                            ui.Button({"ID": 'OkButton', "Text": 'OK'}),
                        ]
                    ),
                ]
            ),
        ]
    )

def show_warning_message(status_tuple):
    use_english = items["LangEnCheckBox"].Checked
    message = status_tuple[0] if use_english else status_tuple[1]
    msgbox.Show()
    msg_items["WarningLabel"].Text = message

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
        "NoGapCheckBox":"字幕之间无间隙" # <-- Add this line
        
    },

    "en": {
        "TitleLabel":"Create subtitles from audio",
        "LangLabel":"Language",
        "ModelLabel":"Model",
        "CreateButton":"Create",
        "HotwordsLabel":"Phrases",
        "MaxCharsLabel":"Max Chars",
        "NoGapCheckBox":"No Gaps Between Subtitles" # <-- Add this line
        
    }
}    

items       = win.GetItems()
msg_items = msgbox.GetItems()

for lang_display_name in LANGUAGE_MAP.keys():
    items["LangCombo"].AddItem(lang_display_name)

whisper_models = [
    "tiny","small","base","medium","large-v3"
]

for model in whisper_models:
    items["ModelCombo"].AddItem(model)
def switch_language(lang):
    """
    根据 lang (可取 'cn' 或 'en') 切换所有控件的文本
    """
    for item_id, text_value in translations[lang].items():
        if item_id in items:
            items[item_id].Text = text_value
        else:
            print(f"[Warning] No control with ID {item_id} exists in items, so the text cannot be set!")

def on_cn_checkbox_clicked(ev):
    items["LangEnCheckBox"].Checked = not items["LangCnCheckBox"].Checked
    if items["LangEnCheckBox"].Checked:
        switch_language("en")
        print("en")
    else:
        print("cn")
        switch_language("cn")
win.On.LangCnCheckBox.Clicked = on_cn_checkbox_clicked

def on_en_checkbox_clicked(ev):
    items["LangCnCheckBox"].Checked = not items["LangEnCheckBox"].Checked
    if items["LangEnCheckBox"].Checked:
        switch_language("en")
        print("en")
    else:
        print("cn")
        switch_language("cn")
win.On.LangEnCheckBox.Clicked = on_en_checkbox_clicked

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
    latest_clip = clips[-1]  # 列表最后一个即刚导入的

    current_media_pool.AppendToTimeline([latest_clip])

    print("🎉 The subtitles were inserted into folder 'srt' and track #", target)
    return True

def find_rendered_file(output_dir: str, custom_name: str) -> Optional[str]:
    """
    在指定目录中查找由渲染任务生成的文件。

    参数:
        output_dir (str): 渲染输出目录。
        custom_name (str): 渲染时设置的自定义名称 (文件名前缀)。

    返回:
        str: 找到的文件的完整路径，如果未找到则返回 None。
    """
    print(f"正在目录 '{output_dir}' 中查找以 '{custom_name}' 开头的文件...")
    
    candidate_files = []
    try:
        # 遍历目录中的所有文件
        for filename in os.listdir(output_dir):
            if filename.startswith(custom_name):
                # 如果文件名以我们设定的前缀开始，则认为它是一个候选文件
                full_path = os.path.join(output_dir, filename)
                candidate_files.append(full_path)
    except FileNotFoundError:
        print(f"错误: 查找目录不存在: {output_dir}")
        return None

    if not candidate_files:
        print("错误: 未找到匹配的渲染文件。")
        return None

    if len(candidate_files) == 1:
        # 如果只有一个匹配项，直接返回
        print(f"成功找到文件: {candidate_files[0]}")
        return candidate_files[0]
    else:
        # 如果有多个匹配项，返回最新创建的那个
        print(f"找到多个匹配文件，将选择最新的一个: {candidate_files}")
        latest_file = max(candidate_files, key=os.path.getctime)
        print(f"选择的文件是: {latest_file}")
        return latest_file
    
def render_timeline_audio(
    output_dir: str,
    custom_name: str,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    audio_codec: str = "aac"
) -> Optional[str]:
    """
    Renders the current timeline's audio using a specific custom name and waits for completion.
    """
    resolve, current_project, _, _, current_timeline, _ = connect_resolve()

    if not current_project:
        print("Error: No project is currently open.")
        return None
    if not current_timeline:
        print("Error: No timeline is currently open.")
        return None

    # Ensure output path exists
    os.makedirs(output_dir, exist_ok=True)

    settings = {
        "SelectAllFrames": True,
        "ExportVideo": False,
        "ExportAudio": True,
        "TargetDir": output_dir,
        "CustomName": custom_name,  # Use the provided custom_name
        "AudioSampleRate": sample_rate,
        "AudioCodec": audio_codec,
        "AudioBitDepth": bit_depth,
    }

    current_project.SetRenderSettings(settings)
    job_id = current_project.AddRenderJob()
    if not job_id:
        print("Error: Failed to add render job.")
        return None

    print(f"Render job added, ID: {job_id}")

    # Start rendering
    if not current_project.StartRendering([job_id], isInteractiveMode=False):
        print("Error: Failed to start rendering.")
        return None
        
    print("Rendering in progress, waiting for completion...")

    while current_project.IsRenderingInProgress():
        time.sleep(2)  # Wait for 2 seconds before checking again

    print("Render complete!")
    
    rendered_filepath = find_rendered_file(output_dir, custom_name)

    return rendered_filepath

def _format_time(seconds: float) -> str:
    """将秒数格式化为 hh:mm:ss,ms 的 SRT 时间戳格式。"""
    milliseconds = int((seconds % 1) * 1000)
    return time.strftime('%H:%M:%S', time.gmtime(seconds)) + f',{milliseconds:03d}'

def _split_segments_by_max_chars(
    segments: Generator[faster_whisper.transcribe.Segment, None, None],
    max_chars: int
) -> List[Dict]:
    """
    根据最大字符数和自然语言标点，将转录片段智能分割成字幕块。

    此版本优化了分割逻辑：
    1. 优先在句子或子句的标点处换行。
    2. 如果在达到max_chars前遇到标点，则提前换行以保证句子完整。
    3. 如果超出max_chars一点但能以标点结尾，则在20%的容差内“拉伸”行。

    Args:
        segments: faster-whisper 返回的带字级时间戳的生成器。
        max_chars: 每个字幕块的最大字符数（硬限制）。

    Returns:
        一个包含字幕块信息的字典列表 (start, end, text)。
    """
    END_OF_CLAUSE_CHARS = tuple(".,?!。，？！")
    subtitle_blocks = []
    current_block = {"start": 0, "end": 0, "text": ""}
    
    # 计算容差后的最大字符数（软限制）
    max_chars_tolerance = int(max_chars * 1.20)

    # 一个辅助函数，用于将当前行添加到最终列表并重置
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
            
            # 如果当前行为空，直接开始新行
            if not current_block["text"]:
                current_block = {"start": word.start, "end": word.end, "text": word_text.lstrip()}
                continue

            # --- 开始智能判断逻辑 ---
            potential_len = len(current_block["text"]) + len(word_text)
            # 判断新加的词是否以标点结尾
            word_ends_clause = word_text.strip().endswith(END_OF_CLAUSE_CHARS)

            # 情况一：新行长度在硬限制内
            if potential_len <= max_chars:
                current_block["text"] += word_text
                current_block["end"] = word.end
                # 如果这是一个自然断点，则立即结束这一行（提前换行）
                if word_ends_clause:
                    finalize_and_reset_block()

            # 情况二：新行长度在容差区域内
            elif potential_len <= max_chars_tolerance:
                # 只有当这个词能构成一个完整子句时，才值得“拉伸”
                if word_ends_clause:
                    current_block["text"] += word_text
                    current_block["end"] = word.end
                    finalize_and_reset_block()
                # 否则，不拉伸。结束当前行，用新词开始下一行
                else:
                    finalize_and_reset_block()
                    current_block = {"start": word.start, "end": word.end, "text": word_text.lstrip()}
            
            # 情况三：新行长度超出容差
            else:
                # 必须换行。结束当前行，用新词开始下一行
                finalize_and_reset_block()
                current_block = {"start": word.start, "end": word.end, "text": word_text.lstrip()}

    # 循环结束后，不要忘记添加最后剩余的行
    finalize_and_reset_block()

    return subtitle_blocks

def _progress_reporter(
    segments_gen,
    total_duration: float,
    callback,
    max_fps: float = 10.0  
):
    """
    • 对 faster-whisper 的 Segment 生成器做包装
    • 每收到一个 Segment 就立即计算进度并回调
    • 通过 max_fps 控制刷新上限，防止极端情况下过度调用
    """
    if total_duration <= 0:          
        for seg in segments_gen:
            yield seg
        callback(100.0)
        return

    last_end       = 0.0
    last_report_ts = 0.0
    min_interval   = 1.0 / max_fps if max_fps > 0 else 0.0

    for seg in segments_gen:
        last_end = max(last_end, seg.end)
        progress = min(last_end / total_duration * 100.0, 100.0)

        now = time.time()
        # —— 仅当距离上次回调已超过 min_interval 才刷新 —— #
        if now - last_report_ts >= min_interval or progress >= 100.0:
            last_report_ts = now
            callback(progress)

        yield seg               

    if progress < 100.0:        
        callback(100.0)

def _remove_gaps_between_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    (最终修正版)
    通过将前一个字幕块的结束时间延长到后一个字幕块的开始时间，来消除间隙。
    """
    if len(blocks) < 2:
        return blocks
    
    # 遍历到倒数第二个元素，因为最后一个元素的结束时间不需要改变
    for i in range(len(blocks) - 1):
        # 将当前字幕块的结束时间，设置为下一个字幕块的开始时间
        blocks[i]["end"] = blocks[i+1]["start"]
        
    return blocks

def generate_srt(
    input_audio: str,
    model_name: str = "base",
    language: Optional[str] = None,
    output_dir: str = ".",
    output_filename: Optional[str] = None,
    max_chars: int = 40,
    batch_size: int = 4, 
    hotwords: Optional[str] = None,  
    verbose: bool = True,
    progress_callback: Optional[callable] = None,
    vad_filter: bool = False,
    remove_gaps: bool = False  # <-- ADD NEW PARAMETER
) -> Optional[str]:
    """
    使用 faster-whisper 转录音频，并生成具有字级时间戳和长度限制的 SRT 字幕文件。
    模型加载失败时，弹窗提示并返回 None。
    """
    # --- 1. 尝试加载模型，失败则弹窗提示 ---
    local_model_path = os.path.join(SCRIPT_PATH, "model", model_name)
    try:
        if verbose:
            show_dynamic_message(f"Loading the model '{model_name}'...", f"正在加载模型 '{model_name}'...")
            print(f"正在加载 faster-whisper 模型 '{model_name}'...")
        model = faster_whisper.WhisperModel(local_model_path)
        pipeline = faster_whisper.BatchedInferencePipeline(model=model)
        if verbose:
            show_dynamic_message(f"Model '{model_name}' loaded successfully.", f"模型 '{model_name}' 加载成功。")
            print(f"模型 '{model_name}' 加载成功。")
    except Exception as e:
        show_dynamic_message(f"Model '{model_name}' is unavailable", f"模型'{model_name}'不可用")
        return None

    # --- 2. 构建转录参数并执行转录 ---
    transcribe_args = {
        "beam_size": 5,
        "log_progress":True,
        "batch_size":batch_size,
        "word_timestamps": True,
        "hotwords": hotwords, 
        "vad_filter": vad_filter
    }
    if language:
        transcribe_args["language"] = language
    if verbose:
        show_dynamic_message(f"[Whisper] Starting ...", f"[Whisper] 开始...")
        print(f"[Whisper] 开始转录：{input_audio}")
        print(transcribe_args)
    segments_gen, info = pipeline.transcribe(input_audio, **transcribe_args)
    show_dynamic_message(f"[Whisper] Language:{info.language}", f"[Whisper] 语言:{info.language}")
    print(info.language)
    # --- 3. 进度回调包装 ---
    if progress_callback:
        segments_gen = _progress_reporter(segments_gen, info.duration, progress_callback)

    # --- 4. 分段控制 ---
    if info.language in ['zh', 'ja', 'th', 'lo', 'km', 'my', 'bo']:
        max_chars = max_chars / 2
    subtitle_blocks = _split_segments_by_max_chars(segments_gen, max_chars)
    # --- NEW: Post-processing to remove gaps if requested ---
    if remove_gaps:
        print("Removing gaps between subtitle blocks...")
        subtitle_blocks = _remove_gaps_between_blocks(subtitle_blocks)

    import re
    for blk in subtitle_blocks:
        # (?<=...)\s+(?=...) 仅匹配两汉字之间的空格
        blk["text"] = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", blk["text"])
    # --- 5. 写入 SRT 文件 ---
    if not output_filename:
        base = os.path.splitext(os.path.basename(input_audio))[0]
        output_filename = f"{base}_whisper"
    os.makedirs(output_dir, exist_ok=True)
    srt_path = os.path.join(output_dir, f"{output_filename}.srt")

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, blk in enumerate(subtitle_blocks, 1):
            f.write(f"{idx}\n")
            f.write(f"{_format_time(blk['start'])} --> {_format_time(blk['end'])}\n")
            f.write(f"{blk['text'].strip()}\n\n")

    if verbose:
        print(f"[Whisper] 已生成 SRT：{srt_path}")
    return srt_path

def on_create_clicked(ev):
    resolve, current_project, _, _, current_timeline, _ = connect_resolve()
    if not current_timeline:
        show_dynamic_message("No active timeline.", "没有激活的时间线。")
        return
        
    timeline_name = current_timeline.GetName()
    
    # Define a consistent file prefix for the timeline's audio
    safe_name = timeline_name.replace(" ", "_")
    audio_file_prefix = f"{safe_name}_audio_temp"

    model_name = items["ModelCombo"].CurrentText
    max_chars  = items["MaxChars"].Value
    display_name = items["LangCombo"].CurrentText

    no_gaps_enabled = items["NoGapCheckBox"].Checked # <-- GET CHECKBOX STATE

    import re
    raw_hotwords = items["Hotwords"].PlainText or ""
    hotwords_list = [
        ph.strip()
        for ph in re.split(r"[，,、；;]\s*|\s+", raw_hotwords)
        if ph.strip()
    ]
    hotwords_str = ",".join(hotwords_list) if hotwords_list else None

    def update_transcribe_progress(progress):
        en = f"Transcribing... {progress:.1f}%"
        zh = f"转录中... {progress:.1f}%"
        show_dynamic_message(en, zh)
    
    try:
        # --- START: New Caching Logic ---
        show_dynamic_message("Checking for cached audio...", "检查音频缓存...")
        print(f"Checking for existing audio file with prefix '{audio_file_prefix}' in '{AUDIO_TEMP_DIR}'...")
        
        # 1. Check if the audio file has already been rendered
        audio_path = find_rendered_file(AUDIO_TEMP_DIR, audio_file_prefix)

        # 2. If not found, render it now
        if not audio_path:
            print("Cached audio not found. Starting new render.")
            show_dynamic_message("Rendering audio...", "音频处理中...")
            audio_path = render_timeline_audio(
                output_dir=AUDIO_TEMP_DIR,
                custom_name=audio_file_prefix
            )
        else:
            print(f"Found cached audio: {audio_path}. Skipping render.")
        # --- END: New Caching Logic ---

        # Determine the output SRT filename
        pattern = os.path.join(SUB_TEMP_DIR, f"{timeline_name}_subtitle_*.srt")
        existing_files = glob.glob(pattern)
        
        indices = []
        for path in existing_files:
            # 从完整路径中提取不带扩展名的文件名 (例如: "MyTimeline_subtitle_1")
            base = os.path.splitext(os.path.basename(path))[0]
            # 按下划线分割
            parts = base.split('_')
            # 获取最后一部分
            idx_str = parts[-1]
            # 检查这部分是否为纯数字
            if idx_str.isdigit():
                indices.append(int(idx_str))

        # 如果找到索引，则取最大值加1，否则从1开始
        next_idx = max(indices) + 1 if indices else 1
        filename = f"{timeline_name}_subtitle_{RAND_CODE}_{next_idx}"

        # 3. Proceed with transcription if the audio path is valid
        if audio_path:
            show_dynamic_message("Transcribing... 0.0%", "转录中... 0.0%")
            resolve.OpenPage("edit")
            srt_path = generate_srt(
                input_audio=audio_path,
                model_name=model_name,
                language=LANGUAGE_MAP.get(display_name),
                output_dir=SUB_TEMP_DIR,
                max_chars=max_chars,
                batch_size=4,
                hotwords=hotwords_str,
                output_filename=filename,
                vad_filter=True,
                progress_callback=update_transcribe_progress,
                remove_gaps=no_gaps_enabled,
            )
            
            if srt_path:
                import_srt_to_first_empty(srt_path)
                show_dynamic_message("Finished! 100%", "转录完成！")
            else:
                # The generate_srt function already shows a model loading error if needed
                print("Failed to generate SRT.")
        else:
            show_dynamic_message("Failed to get audio file.", "获取音频文件失败。")
            
    except Exception as e:
        show_dynamic_message(f"Error: {e}", f"错误: {e}")
        print(f"An error occurred: {e}")
        
win.On.CreateButton.Clicked = on_create_clicked

def on_open_link_button_clicked(ev):
    if items["LangEnCheckBox"].Checked :
        webbrowser.open(SCRIPT_KOFI_URL)
    else :
        webbrowser.open(SCRIPT_BILIBILI_URL)
win.On.CopyrightButton.Clicked = on_open_link_button_clicked

def on_close(ev):
    if os.path.exists(AUDIO_TEMP_DIR):
        try:
            shutil.rmtree(AUDIO_TEMP_DIR)
            print(f"Removed temporary directory: {AUDIO_TEMP_DIR}")
        except OSError as e:
            print(f"Error removing directory {AUDIO_TEMP_DIR}: {e.strerror}")

    if os.path.exists(SUB_TEMP_DIR):
        try:
            shutil.rmtree(SUB_TEMP_DIR)
            print(f"Removed temporary directory: {SUB_TEMP_DIR}")
        except OSError as e:
            print(f"Error removing directory {SUB_TEMP_DIR}: {e.strerror}")
    dispatcher.ExitLoop()
win.On.MyWin.Close = on_close

loading_win.Hide() 
win.Show(); 
dispatcher.RunLoop(); 
win.Hide(); 