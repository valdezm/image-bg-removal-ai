import requests
import sys

# Usage: python client.py input_image.jpg output_image.png

def remove_background(input_path, output_path, server_url="http://localhost:8000/remove-background/"):
    with open(input_path, "rb") as f:
        files = {"file": (input_path, f, "image/jpeg")}
        response = requests.post(server_url, files=files)
        if response.status_code == 200:
            with open(output_path, "wb") as out:
                out.write(response.content)
            print(f"Processed image saved to {output_path}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <input_image> <output_image>")
        sys.exit(1)
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    remove_background(input_image, output_image)
