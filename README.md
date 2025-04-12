# FUNCAP Image Matching API

Supports multiple FunCaptcha variants, loads models smart so it’s quick, and gives you back the best match with no fuss.

## Features

- Fast image similarity matching using ONNX Runtime
- Support for multiple variants/models
- Simple REST API interface
- Cross-origin resource sharing (CORS) enabled

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ziad-gg/fcap-classification.git
   cd fcap-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the proper model files in the `models` directory:
   ```
   mkdir -p models
   # Place your .onnx model files in the models directory
   ```

## Usage

1. Start the Flask server:
   ```
   python server.py
   ```

2. Send a POST request to `/match_image` endpoint with:
   - `image`: Base64-encoded image data
   - `variant`: The variant/model to use for matching

### Example Request

```json
{
  "image": "base64_encoded_image_data_here",
  "variant": "waterIconCup"
}
```

### Example Response

```json
{
  "success": true,
  "result": {
    "best_match_index": 2,
    "similarity_score": 0.87
  }
}
```

## Supported Variants

| Variant Name | Description | Available |
|--------------|-------------|-----------|
| waterIconCup | WaterIconCup model | ✅ Basic version |

## Premium Models

For higher accuracy models and additional variants, contact **@ziaath** on Telegram. Premium models offer:

- Higher accuracy rates
- Faster processing
- Support for more image types
- Custom training for your specific use cases

## API Reference

### POST /match_image

Matches a reference image against potential matches using the specified model variant.

**Parameters:**
- `image` (required): Base64-encoded image data
- `variant` (optional): Model variant to use (defaults to "waterIconCup")

**Responses:**
- 200: Successful operation with match results
- 400: Bad request (missing parameters)
- 500: Server error

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.