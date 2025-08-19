import sys
import os
import argparse
import time

# Add local packages to Python path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'video-subtitle-remover'))

from backend.main import SubtitleRemover
from video_tran.subtitle_processor import SubtitleProcessor

def main():
    parser = argparse.ArgumentParser(description="Remove subtitles from a video dynamically using a subtitle file.")
    parser.add_argument('--video_path', type=str, default=r"D:\\Python\\Project\\VideoTran\\videos\\223.mp4", help='Path to the input video file.')
    parser.add_argument('--subtitle_path', type=str, default=r"D:\\Python\\Project\\VideoTran\\videos\\223.json", help='Path to the JSON subtitle file, which will be enriched with bounding box data.')
    parser.add_argument('--output_path', type=str, default=r"D:\\Python\\Project\\VideoTran\\videos\\223_no_subtitle.mp4", help='Path to the output video file.')
    parser.add_argument('--force-detect', action='store_true', help='Force re-detection of subtitle bounding boxes, ignoring any cached data in the JSON file.')

    args = parser.parse_args()

    # --- Step 1: Process subtitle file to get bounding boxes ---
    print("--- Step 1: Processing subtitle file to ensure bounding boxes are present ---")
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    processor = SubtitleProcessor(
        video_path=args.video_path,
        subtitle_path=args.subtitle_path,
        workspace_dir=workspace_dir
    )
    enriched_subtitle_path = processor.run(force_run=args.force_detect)

    if not enriched_subtitle_path:
        print("Error: Subtitle processing failed. Cannot proceed with removal.")
        return

    print("\n--- Step 2: Starting dynamic subtitle removal ---")
    
    # --- Step 2: Call the backend to remove subtitles dynamically ---
    remover = SubtitleRemover(
        video_path=args.video_path, 
        subtitle_json_path=enriched_subtitle_path,
        output_path=args.output_path
    )
    remover.run()

    print(f"\nSubtitle removal complete. Output saved to: {args.output_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")