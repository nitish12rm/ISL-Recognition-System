from fastapi import FastAPI, File, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
import os
import numpy as np
import uvicorn
import ollama  # Ollama Python API for on-device LLM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from helper_functions import convert_video_to_pose_embedded_np_array

app = FastAPI()
actions = np.array(["Hello", "How are you", "Thank you"])


def initialize_model():
    """
    Initializes LSTM model and loads the trained model weights
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights("lstm-model/170-0.83.hdf5")
    return model


# Initialize the model globally
model = initialize_model()


def generate_sentence_with_ollama(predicted_words: list):
    """
    Uses the on-device Ollama Llama 3 model to generate a contextual sentence.
    Ensures the sentence only contains the given words.
    """
    words_str = ", ".join(predicted_words)
    prompt = f"Create a meaningful sentence using only these words: {words_str}. Do not add new words."

    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        generated_sentence = response.get("message", {}).get("content", "").strip()
        return generated_sentence if generated_sentence else "No valid sentence generated."
    except Exception as e:
        return f"Error generating sentence: {str(e)}"


@app.post("/upload-videos/")
async def upload_videos(files: list[UploadFile] = File(...)):
    """
    Uploads multiple video files, processes them, and predicts the actions.
    Then generates a contextual sentence using only the predicted words.
    """
    predicted_words = []

    for file in files:
        video_format = os.path.splitext(file.filename)[1].lower()

        if video_format not in ['.mp4', '.avi', '.mov']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format for {file.filename}. Accepted formats: .mp4, .avi, .mov"
            )

        temp_path = None
        try:
            # Create a temporary file
            with NamedTemporaryFile(suffix=video_format, delete=False) as temp:
                contents = await file.read()
                temp.write(contents)
                temp_path = temp.name

            # Process the video and convert it to a pose-embedded numpy array
            out_np_array = convert_video_to_pose_embedded_np_array(temp_path, remove_input=False)

            # Validate input shape for the model
            if out_np_array.shape != (45, 258):
                raise ValueError(
                    f"Unexpected input shape: {out_np_array.shape}. Expected: (45, 258)."
                )

            # Make prediction
            prediction = model.predict(np.expand_dims(out_np_array, axis=0))
            predicted_action = actions[np.argmax(prediction, axis=1)[0]]

            predicted_words.append(predicted_action)

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # Generate contextual sentence using only the predicted words
    contextual_sentence = generate_sentence_with_ollama(predicted_words)

    return {
        "status": "success",
        "predicted_words": predicted_words,
        "contextual_sentence": contextual_sentence
    }


@app.get("/test/")
async def test():
    return {"status": "healthy", "message": "API is working"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
