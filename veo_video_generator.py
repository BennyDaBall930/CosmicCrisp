
import os
import sys
import time
import argparse
from google import genai
from google.genai import types

def generate_video(prompt: str, output_path: str, model: str):
    """
    Generates a video using the Veo API based on a prompt.

    Args:
        prompt (str): The text prompt describing the video.
        output_path (str): The path to save the generated MP4 file.
        model (str): The model ID to use for generation.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        # genai.configure() is not needed when using Client() which reads from env var
        client = genai.Client()

        print(f"Submitting video generation request with prompt: '{prompt}'")
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                resolution="720p"
            ),
        )

        print("Request submitted. Pol        pcompletion (this may take a minute)...")
        while not operation.done:
            print("Waiting for video generation to complete...")
            time.sleep(10)
            operation = client.operations.get(operation)

        print("Video generation complete.")
        generated_video = operation.response.generated_videos[0]
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print('Downloading video file...')
        client.files.download(file=generated_video.video)
        generated_video.video.save(output_path)
        print(f"\n[SUCCESS] Video saved successfully to: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate a video using the Gemini Veo API.")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="The text prompt for the video.")
    parser.add_argument("-o", "--output", type=str, default="tmp/generated_video.mp4", help="The output path for the video file.")
    parser.add_argument("-m", "--model", type=str, default="veo-3.0-fast-generate-001", help="The model to use (e.g., veo-3.0-fast-generate-001).")
    args = parser.parse_args()

    generate_video(args.prompt, args.output, args.model)

if __name__ == "__main__":
    main()

