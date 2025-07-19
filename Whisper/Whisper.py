SCRIPT_NAME    = "Whisper "
SCRIPT_VERSION = "1.0"
SCRIPT_AUTHOR  = "HEIBA"

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
WINDOW_WIDTH, WINDOW_HEIGHT = 300, 300
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
from typing import Optional, List, Generator

SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
AUDIO_TEMP_DIR = os.path.join(SCRIPT_PATH, "audio_temp")
SUB_TEMP_DIR = os.path.join(SCRIPT_PATH, "sub_temp")
RAND_CODE = "".join(random.choices(string.digits, k=2))

if not hasattr(sys.stderr, "flush"):
    sys.stderr.flush = lambda: None

try:
    import faster_whisper
except ImportError:
    # 根据不同平台设置 Lib 目录为绝对路径
    system = platform.system()
    if system == "Windows":
        program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        lib_dir = os.path.join(
            program_data,
            "Blackmagic Design",
            "DaVinci Resolve",
            "Fusion",
            "HB",
            "Whisper",
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
            "Whisper",
            "Lib"
        )
    else:
        lib_dir = os.path.normpath(
            os.path.join(SCRIPT_PATH, "..", "..", "..", "HB", "Whisper", "Lib")
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

ui       = fusion.UIManager
dispatcher = bmd.UIDispatcher(ui)
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

                # ===== 4.1 翻译页 =====
                ui.VGroup({"Weight":1},[
                    ui.Label({"ID":"TitleLabel","Text":"从音频创建字幕","Alignment": {"AlignHCenter": True, "AlignVCenter": True},"Weight":0.1}),
                    ui.HGroup({"Weight":0.1},[
                        ui.Label({"ID":"ModelLabel","Text":"模型","Weight":0.1}),
                        ui.ComboBox({"ID":"ModelCombo","Weight":0.1}),
                    ]),
                    ui.HGroup({"Weight":0.1},[
                        ui.Label({"ID":"LangLabel","Text":"语言","Weight":0.1}),
                        ui.ComboBox({"ID":"LangCombo","Weight":0.1}),
                    ]),
                    ui.HGroup({"Weight":0.1},[
                        ui.Label({"ID":"MaxCharsLabel","Text":"最大值","Weight":0.1}),
                        ui.LineEdit({"ID":"MaxChars","Text":"42","ReadOnly":False,"Weight":0.1}),
                    ]),
                    ui.Button({"ID":"CreateButton","Text":"创建","Weight":0.15}),
                    ui.Label({"ID": "StatusLabel", "Text": " ", "Alignment": {"AlignHCenter": True, "AlignVCenter": True},"Weight":0.1}),
                    ui.HGroup({"Weight":0.1},[
                        ui.CheckBox({"ID":"LangEnCheckBox","Text":"EN","Checked":True,"Weight":0}),
                        ui.CheckBox({"ID":"LangCnCheckBox","Text":"简体中文","Checked":False,"Weight":1}),
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
    # 元组索引 0 为英文，1 为中文
    message = status_tuple[0] if use_english else status_tuple[1]
    msgbox.Show()
    msg_items["WarningLabel"].Text = message

def show_dynamic_message(en_text, zh_text):
    """直接弹窗显示任意中英文文本的动态消息"""
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
        "MaxCharsLabel":"每行最大字符",
        
    },

    "en": {
        "TitleLabel":"Create subtitles from audio",
        "LangLabel":"Target Language",
        "ModelLabel":"Model",
        "CreateButton":"Translate",
        "MaxCharsLabel":"Max Chars",
        
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
    # 缓存复选框状态
    checked = items["LangEnCheckBox"].Checked

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

# 连接 DaVinci Resolve
def connect_resolve():
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    media_pool = project.GetMediaPool(); 
    root_folder = media_pool.GetRootFolder()
    timeline      = project.GetCurrentTimeline()
    fps     = float(project.GetSetting("timelineFrameRate"))
    return resolve, project, media_pool,root_folder,timeline, fps

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
    output_dir: str = "./exports",
    sample_rate: int = 48000,
    bit_depth: int = 16,
    audio_codec: str = "aac" # 知识库中 "aac" 是示例, 对于WAV, pcm更合适
) -> Optional[str]:
    """
    渲染当前时间线的音频并导出为 WAV 文件，并等待渲染完成。
    """
    resolve, current_project, current_media_pool, current_root_folder, current_timeline, fps = connect_resolve()

    if not current_project:
        print("错误: 未打开任何项目")
        return None
    if not current_timeline:
        print("错误: 未打开任何时间线")
        return None

    # 使用时间线名称生成音频文件名
    timeline_name = current_timeline.GetName() # 
    safe_name = timeline_name.replace(" ", "_")
    
    custom_name = f"{safe_name}_audio"

    # 确保输出路径存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置渲染参数（仅导出音频）
    # 注意: 知识库中没有明确指出 "Format":"WAV" 的用法，但这是行业标准。
    # "AudioCodec" 对于 WAV 格式，通常是未压缩的 PCM。
    # "aac" 更适用于 MP4 等容器。这里我们假定 API 内部能处理。
    settings = {
        "SelectAllFrames": True,
        "ExportVideo": False, 
        "ExportAudio": True, 
        "TargetDir": output_dir, 
        "CustomName": custom_name, 
        "AudioSampleRate": sample_rate, 
        "AudioCodec": audio_codec,
        "AudioBitDepth": bit_depth, 
    }

    # 应用渲染设置并添加渲染任务
    current_project.SetRenderSettings(settings) 
    job_id = current_project.AddRenderJob() 
    if not job_id:
        print("错误: 渲染任务添加失败")
        return None

    print(f"渲染任务已添加, Job ID: {job_id}")

    # 开始渲染
    if not current_project.StartRendering([job_id],isInteractiveMode=False): # [cite: 97]
        print("错误: 渲染启动失败")
        return None
        
    print("渲染已启动，正在等待完成...")

    # --- 新增的等待逻辑 ---
    # 循环检查渲染是否仍在进行中
    while current_project.IsRenderingInProgress(): # 
        # 打印进度，让用户知道没有卡死
        # GetRenderJobStatus可以提供更详细的进度，这里用简单的等待
        print("渲染进行中...")
        time.sleep(2)  # 每2秒检查一次，避免CPU占用过高

    print("渲染完成!")
    
    rendered_filepath = find_rendered_file(output_dir, custom_name)

    return rendered_filepath
def _format_time(seconds: float) -> str:
    """将秒数格式化为 hh:mm:ss,ms 的 SRT 时间戳格式。"""
    milliseconds = int((seconds % 1) * 1000)
    return time.strftime('%H:%M:%S', time.gmtime(seconds)) + f',{milliseconds:03d}'

def _split_segments_by_max_chars(
    segments: Generator[faster_whisper.transcribe.Segment, None, None],
    max_chars: int
) -> List[dict]:
    """
    根据最大字符数将转录的片段分割成字幕块。

    Args:
        segments: faster-whisper 返回的带字级时间戳的生成器。
        max_chars: 每个字幕块的最大字符数。

    Returns:
        一个包含字幕块信息的字典列表 (start, end, text)。
    """
    subtitle_blocks = []
    current_block = {"start": 0, "end": 0, "text": ""}

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            # 检查添加新单词后是否会超过长度限制
            if len(current_block["text"]) + len(word.word) + 1 > max_chars and current_block["text"]:
                # 如果超过限制，则保存当前块并开始新块
                subtitle_blocks.append(current_block)
                current_block = {"start": word.start, "end": word.end, "text": word.word}
            else:
                # 否则，将单词添加到当前块
                if not current_block["text"]:
                    # 这是新块的第一个单词
                    current_block["start"] = word.start
                    current_block["text"] = word.word
                else:
                    # 在单词前添加一个空格
                    current_block["text"] += " " + word.word
                current_block["end"] = word.end

    # 添加最后一个字幕块
    if current_block["text"]:
        subtitle_blocks.append(current_block)

    return subtitle_blocks

def _progress_reporter(
    segments_gen,
    total_duration: float,
    callback,
    max_fps: float = 10.0  # 最多每秒刷新 10 次；设为 0 表示无节流
):
    """
    • 对 faster-whisper 的 Segment 生成器做包装
    • 每收到一个 Segment 就立即计算进度并回调
    • 通过 max_fps 控制刷新上限，防止极端情况下过度调用
    """
    if total_duration <= 0:          # 极端兜底
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

        yield seg               # 继续向下游传递

    if progress < 100.0:        # 收尾兜底
        callback(100.0)

def generate_srt(
    input_audio: str,
    model_name: str = "base",
    language: Optional[str] = None,
    output_dir: str = ".",
    output_filename: Optional[str] = None,
    max_chars: int = 40,  # 新增参数，控制每块字幕的最大字符数
    verbose: bool = True,
    progress_callback: Optional[callable] = None
) -> Optional[str]:
    """
    使用 faster-whisper 转录音频，并生成具有字级时间戳和长度限制的 SRT 字幕文件。

    Args:
        input_audio: 输入的音频文件路径。
        model_name: 使用的 whisper 模型名称 (例如, "base", "small", "large-v3")。
        output_dir: SRT 文件的输出目录。
        output_filename: 输出文件的基本名称 (不含扩展名)。
        max_chars: 每个字幕块的最大字符数。
        verbose: 是否打印详细日志。
        progress_callback: 用于报告进度的回调函数。

    Returns:
        生成的 SRT 文件的路径，如果失败则返回 None。
    """
    try:
        # --- 1. 加载 faster-whisper 模型 ---
        if verbose:
            print(f"正在加载 faster-whisper 模型 '{model_name}'...")
        local_model_path = os.path.join(SCRIPT_PATH, "model", model_name)
        model = faster_whisper.WhisperModel(local_model_path)
        if verbose:
            print(f"模型 '{model_name}' 加载成功。")

        transcribe_args = {"beam_size": 5, "word_timestamps": True}
        if language:
            transcribe_args["language"] = language
           

        if verbose:
            print(f"[Whisper] 开始转录：{input_audio}")

        segments_gen, info = model.transcribe(input_audio, **transcribe_args)

        if verbose:
            print(f"[Whisper] language:{info.language}")
        # 3. 包装生成器，按改良算法汇报进度
        if progress_callback:
            segments_gen = _progress_reporter(segments_gen, info.duration, progress_callback)

        # 4. 按行长重新组合字幕块
        subtitle_blocks = _split_segments_by_max_chars(segments_gen, max_chars)

        # 5. 生成 SRT 文件
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

    except Exception as e:
        print(f"[Whisper] 生成 SRT 失败：{e}")
        return None

def on_create_clicked(ev):
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d%H%M")
    model_name = items["ModelCombo"].CurrentText
    max_chars  = int(items["MaxChars"].Text)
    display_name = items["LangCombo"].CurrentText
    status_label = items["StatusLabel"]

    def update_transcribe_progress(progress):
        en = f"Transcribing... {progress:.1f}%"
        zh = f"转录中... {progress:.1f}%"
        show_dynamic_message(en, zh)
        #status_label.Text = f"Transcribing... {progress:.1f}%"


    try:
        show_dynamic_message("Rendering audio...","音频处理... ")
        audio_path = render_timeline_audio(output_dir=AUDIO_TEMP_DIR)
        filename = f"{timestamp}_{RAND_CODE}.srt"

        if audio_path:
            show_dynamic_message("Transcribing... 0.0% ","转录中... 0.0% ")

            resolve.OpenPage("edit")
            srt_path = generate_srt(
                input_audio=audio_path,
                model_name=model_name,
                language=LANGUAGE_MAP.get(display_name),
                output_dir=SUB_TEMP_DIR,
                max_chars=max_chars,
                output_filename=filename,
                progress_callback=update_transcribe_progress
            )
            
            if srt_path:
                import_srt_to_first_empty(srt_path)
                show_dynamic_message("Finished! 100% ","转录完成！")
            else:
                show_dynamic_message("Failed to generate SRT! ","转录失败！ ")
        else:
            show_dynamic_message("Failed to render audio! ","渲染失败！ ")
    except Exception as e:
        status_label.Text = f"Error: {e}"
        print(f"An error occurred: {e}")
        
win.On.CreateButton.Clicked = on_create_clicked

def on_open_link_button_clicked(ev):
    if items["LangEnCheckBox"].Checked :
        webbrowser.open(SCRIPT_KOFI_URL)
    else :
        webbrowser.open(SCRIPT_BILIBILI_URL)
win.On.CopyrightButton.Clicked = on_open_link_button_clicked

def on_close(ev):
    # Clean up temporary directories
    audio_temp_dir = os.path.join(SCRIPT_PATH, "audio_temp")
    sub_temp_dir = os.path.join(SCRIPT_PATH, "sub_temp")
    
    if os.path.exists(audio_temp_dir):
        try:
            shutil.rmtree(audio_temp_dir)
            print(f"Removed temporary directory: {audio_temp_dir}")
        except OSError as e:
            print(f"Error removing directory {audio_temp_dir}: {e.strerror}")

    if os.path.exists(sub_temp_dir):
        try:
            shutil.rmtree(sub_temp_dir)
            print(f"Removed temporary directory: {sub_temp_dir}")
        except OSError as e:
            print(f"Error removing directory {sub_temp_dir}: {e.strerror}")
            
    dispatcher.ExitLoop()
win.On.MyWin.Close = on_close

win.Show(); 
dispatcher.RunLoop(); 
win.Hide(); 