import os
import face_recognition
from flask import Flask, jsonify, request
from PIL import Image, ExifTags
import io

# Initialize the Flask app
app = Flask(__name__)


# Define the route
@app.route('/scanFace', methods=['POST'])
def scan_face():
    try:
        # Check if the file is uploaded
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        uploaded_file = request.files['file']

        # If the file is empty, return an error
        if uploaded_file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', 'snout.jpg')
        uploaded_file.save(file_path)
        print(f"File saved at: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        # Rotate the image if necessary
        image = Image.open(file_path)

        # Correct the image orientation based on EXIF data
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None:
                for orientation_tag in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation_tag] == 'Orientation':
                        break
                orientation = exif.get(orientation_tag, None)

                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)

        # Save the corrected image back to the file path
        image.save(file_path)

        # Load the uploaded image for face recognition
        try:
            uploaded_image = face_recognition.load_image_file(file_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return jsonify({"message": "Error loading the uploaded image."}), 500

        # Get the face encodings of the uploaded image
        try:
            uploaded_encodings = face_recognition.face_encodings(uploaded_image)
            print(f"Detected {len(uploaded_encodings)} face(s) in the uploaded image.")
        except Exception as e:
            print(f"Error during face encoding: {e}")
            return jsonify({"message": "Error processing the face in the uploaded image."}), 500

        if not uploaded_encodings:
            return jsonify({"message": "No face detected in the uploaded image"}), 400

        # Load and compare with images in 'human' directory
        human_folder = "human"  # Path to the directory with stored human images
        human_images = os.listdir(human_folder)

        for image_name in human_images:
            image_path = os.path.join(human_folder, image_name)
            print(f"Loading human image: {image_path}")

            try:
                human_image = face_recognition.load_image_file(image_path)
                human_encodings = face_recognition.face_encodings(human_image)
            except Exception as e:
                print(f"Error processing human image {image_name}: {e}")
                continue  # Skip this human image if there's an error loading it

            if not human_encodings:
                print(f"No face found in human image: {image_name}")
            else:
                print(f"Face found in human image: {image_name}")

                # Debugging: print the human encodings for this image
                print(f"Human encodings for {image_name}: {human_encodings}")

                # Compare faces using face_recognition.compare_faces
                for uploaded_encoding in uploaded_encodings:
                    try:
                        print(f"Comparing uploaded encoding to human encodings for image: {image_name}")
                        matches = face_recognition.compare_faces(human_encodings, uploaded_encoding)

                        # Debugging: print the match result
                        print(f"Match result: {matches}")

                    except Exception as e:
                        print(f"Error comparing faces: {e}")
                        continue  # Skip comparison if there is an error

                    if True in matches:
                        print(f"Face matches a known human in {image_name}")
                        return jsonify({"message": "Face matches a known human!"}), 200

        # No match found
        return jsonify({"message": "No match found."}), 200

    except Exception as e:
        print(f"Error during face recognition: {e}")  # Log the specific error
        return jsonify({"message": f"An error occurred while processing the image. {e}"}), 500


if __name__ == '__main__':
    # Run the app on the desired host and port with debug mode enabled
    app.run(host='0.0.0.0', port=5001, debug=True)
