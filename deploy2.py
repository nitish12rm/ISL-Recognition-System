from fastapi import FastAPI, File, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
import os
import numpy as np
import uvicorn
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


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a video file, processes it, and predicts the action made in the video.
    Returns: Predicted action.
    """
    video_format = os.path.splitext(file.filename)[1].lower()
    print(f"Filename: {file.filename}, Content Type: {file.content_type}")

    # Validate video format
    if video_format not in ['.mp4', '.avi', '.mov']:
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Accepted formats: .mp4, .avi, .mov"
        )

    temp_path = None
    try:
        # Create a temporary file
        with NamedTemporaryFile(suffix=video_format, delete=False) as temp:
            contents = await file.read()  # Use await for async file reading
            temp.write(contents)
            temp_path = temp.name

        print(f"Temporary file created: {temp_path}")

        # Process the video and convert it to a pose-embedded numpy array
        out_np_array = convert_video_to_pose_embedded_np_array(temp_path, remove_input=False)
        print(f"Processed numpy array shape: {out_np_array.shape}")

        # Validate input shape for the model
        if out_np_array.shape != (45, 258):
            raise ValueError(
                f"Unexpected input shape: {out_np_array.shape}. Expected: (45, 258)."
            )

        # Make prediction
        prediction = model.predict(np.expand_dims(out_np_array, axis=0))
        predicted_action = actions[np.argmax(prediction, axis=1)[0]]

        return {
            "status": "success",
            "action": predicted_action,
            "confidence": np.float64 (np.max(prediction))  # Convert to float for JSON serialization
        }

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file deleted: {temp_path}")


@app.get("/test/")
async def test():
    return {"status": "healthy", "message": "API is working"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)