import os
import json
import ffmpeg
import cv2
import paddle
from paddleocr import PaddleOCR
from thefuzz import fuzz
import time

class SubtitleProcessor:
    """
    Processes a subtitle file to detect and cache bounding boxes for each subtitle entry.
    """
    def __init__(self, video_path, subtitle_path, workspace_dir):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.workspace_dir = workspace_dir
        
        self.frames_dir = os.path.join(self.workspace_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        print(f"[Processor] Workspace for frames: {os.path.abspath(self.frames_dir)}")

    def _time_str_to_ms(self, time_str):
        """Converts HH:MM:SS,ms or MM:SS,ms time string to milliseconds."""
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s_ms = parts
        elif len(parts) == 2:
            h = 0
            m, s_ms = parts
        else:
            raise ValueError(f"Time format in subtitle file is not supported: {time_str}")
        
        s, ms = s_ms.split(',')
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    def run(self, force_run=False):
        """
        Main method to process and enrich the subtitle file.
        Iterates through each subtitle, detects its bounding box if not already present,
        and saves the updated data back to the subtitle file.
        """
        print("[Processor] Starting subtitle processing.")

        # 1. Load subtitle file
        try:
            with open(self.subtitle_path, 'r', encoding='utf-8') as f:
                subs_data = json.load(f)
        except Exception as e:
            print(f"[Processor] Error reading or parsing subtitle file: {e}")
            return None

        # 2. Get video resolution
        try:
            probe = ffmpeg.probe(self.video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            video_height = int(video_stream['height'])
            video_width = int(video_stream['width'])
        except ffmpeg.Error as e:
            print(f"[Processor] Error: Could not get video resolution: {e.stderr.decode()}")
            return None

        # 3. Initialize PaddleOCR and check for GPU
        print("[Processor] Loading PaddleOCR model into memory (this may take a moment)...")
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, show_log=False)
            
            # Check if Paddle is actually using GPU
            if 'cpu' in paddle.get_device():
                print("\n[Processor] FATAL ERROR: PaddleOCR failed to initialize on GPU and has fallen back to CPU.")
                print("     Please check your CUDA and cuDNN installation.")
                print("     Halting execution as per user request to avoid long processing time on CPU.")
                return None
            else:
                print(f"[Processor] Successfully initialized PaddleOCR on device: {paddle.get_device()}")

        except Exception as e:
            print(f"[Processor] Error: Failed to initialize PaddleOCR: {e}")
            return None

        # 4. Process each subtitle entry
        for i, sub in enumerate(subs_data):
            print(f"--- Processing subtitle {i+1}/{len(subs_data)}: \"{sub.get('text')}\" ---")

            if not force_run and 'bbox' in sub:
                print("  - Bounding box already exists in cache. Skipping.")
                continue

            try:
                start_ms = self._time_str_to_ms(sub['startTime'])
                end_ms = self._time_str_to_ms(sub['endTime'])
                mid_point_s = (start_ms + (end_ms - start_ms) / 2) / 1000.0
                known_text = sub['text']

                frame_filename = f"frame_at_{sub['startTime'].replace(':', '-').replace(',', '_')}.png"
                frame_filepath = os.path.join(self.frames_dir, frame_filename)

                ffmpeg.input(self.video_path, ss=mid_point_s).output(frame_filepath, vframes=1, loglevel="quiet").run(overwrite_output=True)
                
                results = ocr_engine.ocr(frame_filepath, cls=True)
                if not results or not results[0]:
                    print("  - OCR found no text in this frame.")
                    continue

                best_match = None
                highest_score = -1
                for (bbox, (ocr_text, conf)) in results[0]:
                    similarity_score = fuzz.ratio(known_text, ocr_text)
                    if similarity_score > highest_score:
                        highest_score = similarity_score
                        best_match = (bbox, ocr_text, conf)
                
                if best_match and highest_score > 30:
                    ocr_bbox, ocr_text, _ = best_match
                    print(f'  - [Found Best Match] Score: {highest_score}, Box: {ocr_bbox}, Text: "{ocr_text}"')

                    # --- FIX for Smart Expansion Logic ---
                    xmin = min(p[0] for p in ocr_bbox)
                    xmax = max(p[0] for p in ocr_bbox)
                    ymin = min(p[1] for p in ocr_bbox)
                    ymax = max(p[1] for p in ocr_bbox)

                    if len(ocr_text) > 0 and len(ocr_text) < len(known_text):
                        avg_char_width = (xmax - xmin) / len(ocr_text)
                        missing_chars = len(known_text) - len(ocr_text)
                        expand_width = (missing_chars * avg_char_width) / 2
                        xmin -= expand_width
                        xmax += expand_width

                    # Apply padding and ensure bounds are within video dimensions
                    padding_y = 3
                    padding_x = 5 # Add small horizontal padding for safety
                    final_xmin = max(0, int(xmin - padding_x))
                    final_xmax = min(video_width, int(xmax + padding_x))
                    final_ymin = max(0, int(ymin - padding_y))
                    final_ymax = min(video_height, int(ymax + padding_y))

                    final_bbox = [[final_xmin, final_ymin], [final_xmax, final_ymin], 
                                  [final_xmax, final_ymax], [final_xmin, final_ymax]]
                    
                    sub['bbox'] = final_bbox
                    print(f"  - Stored BBox with smart expansion: {final_bbox}")
                else:
                    print("  - [No suitable match found for this frame]")

            except Exception as e:
                print(f"[Processor] Error: An error occurred while processing frame for subtitle '{sub.get('text')}': {e}")
                continue
        
        # 5. Finalize and save
        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            json.dump(subs_data, f, indent=4, ensure_ascii=False)
        print(f"[Processor] Subtitle processing complete. Enriched data saved to {self.subtitle_path}")
        
        return self.subtitle_path